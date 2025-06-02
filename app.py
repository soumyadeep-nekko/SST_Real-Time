import os
import time
import cv2
import threading  # NEW
from flask import Flask, Response, render_template_string, stream_with_context
from ultralytics import YOLO

# Fix OpenMP duplicate library error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO('yolov8m.pt')

rtsp_url = 'rtsp://admin:AGRIsal@agrisal-blue-lx.dyndns.info:40080/cam/realmonitor?channel=2&subtype=0'

person_class_id = 0
cell_phone_class_id = 67
vehicle_class_ids = [1, 2, 3, 5, 7]

class_names = model.names
tracked_vehicles = {}

DWELL_TIME = 60
WARNING_TIME = 45
MOVE_THRESHOLD = 40

event_log = []  # NEW: store latest events (rolling list)
MAX_EVENTS = 20  # NEW: max items in log

inference_log = []  # NEW: Store inference results per frame
MAX_INFERENCES = 20  # Limit to avoid overflow


def generate_frames():
    results = model.track(
        source=rtsp_url,
        show=False,
        stream=True,
        tracker="bytetrack.yaml",
        persist=True
    )

    for result in results:
        frame = result.orig_img
        current_time = time.time()

        if result.boxes.id is None:
            continue

        ids = result.boxes.id.cpu().numpy().astype(int)
        classes = result.boxes.cls.cpu().numpy().astype(int)
        boxes = result.boxes.xyxy.cpu().numpy()

        person_count = 0
        vehicle_count = 0

        persons = []
        cell_phones = []

        for track_id, cls, box in zip(ids, classes, boxes):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            label = class_names[cls]

            if cls == person_class_id:
                person_count += 1
                persons.append((track_id, (x1, y1, x2, y2)))
            elif cls == cell_phone_class_id:
                cell_phones.append((track_id, (x1, y1, x2, y2)))
            elif cls in vehicle_class_ids:
                vehicle_count += 1

            if cls not in vehicle_class_ids:
                continue

            if track_id not in tracked_vehicles:
                tracked_vehicles[track_id] = {'start_time': current_time, 'bbox': (cx, cy)}
            else:
                prev_cx, prev_cy = tracked_vehicles[track_id]['bbox']
                distance = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5

                if distance > MOVE_THRESHOLD:
                    tracked_vehicles[track_id] = {'start_time': current_time, 'bbox': (cx, cy)}

            dwell_duration = current_time - tracked_vehicles[track_id]['start_time']

            if dwell_duration >= DWELL_TIME:
                color = (0, 0, 255)
                alert_msg = f'ALERT: {label} {track_id} idle {int(dwell_duration)}s'
                cv2.putText(frame, alert_msg, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                print(alert_msg)
                event_log.append(alert_msg)  # NEW
            elif dwell_duration >= WARNING_TIME:
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label} ID: {track_id}', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for pid, p_box in persons:
            px1, py1, px2, py2 = p_box
            for cid, c_box in cell_phones:
                cx1, cy1, cx2, cy2 = c_box
                if (px1 < cx2 and px2 > cx1 and py1 < cy2 and py2 > cy1):
                    color = (255, 0, 0)
                    alert_msg = f'ALERT: Person {pid} using mobile phone'
                    cv2.putText(frame, alert_msg, (px1, py1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    print(alert_msg)
                    cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
                    event_log.append(alert_msg)  # NEW

        while len(event_log) > MAX_EVENTS:  # NEW: keep log bounded
            event_log.pop(0)

        text_color = (50, 50, 50)
        cv2.putText(frame, f"People: {person_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        # NEW: Log inference classes for this frame
        frame_summary = []
        for cls in classes:
            frame_summary.append(class_names[cls])
        summary_text = f"{len(frame_summary)}: " + ', '.join(frame_summary)
        summary_text += f", {result.speed['inference']:.1f}ms"
        print(summary_text)  # You already have this line from console log
        inference_log.append(summary_text)
        while len(inference_log) > MAX_INFERENCES:
            inference_log.pop(0)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    # Simple HTML page showing video and events
    return render_template_string('''
    <html>
<head>
    <title>SST Vision</title>
    <style>
        body { font-family: sans-serif; margin: 0; padding: 0; background: #f0f0f0; }
        #video-container { text-align: center; margin-top: 20px; }
        .box { margin: 20px auto; width: 90%; max-width: 800px; background: #fff; padding: 10px; border-radius: 10px; height: 150px; overflow-y: auto; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
    </style>
</head>
<style>
    .green-dot::marker {
        color: green;
    }
    .red-dot::marker {
        color: red;
    }
    .yellow-dot::marker {
        color: orange;
    }
</style>
<body>
    <div id="video-container">
        <h2>Live Camera Stream</h2>
        <img src="/stream" style="max-width: 100%; border: 4px solid #333;" />
    </div>

    <div class="box">
        <h4>Detected Events:</h4>
        <ul id="event-list"></ul>
    </div>

    <div class="box">
        <h4>Inference Log:</h4>
        <ul id="inference-list"></ul>
    </div>

    <script>
    const eventList = document.getElementById('event-list');
    const inferenceList = document.getElementById('inference-list');

    const evtSource = new EventSource("/events");
    evtSource.onmessage = function(event) {
        const li = document.createElement("li");
        li.textContent = event.data;

        // 🔴 Add logic to color events
        if (event.data.toLowerCase().includes("alert")) {
            li.classList.add("red-dot");
        } else if (event.data.toLowerCase().includes("warning")) {
            li.classList.add("yellow-dot");
        } else {
            li.classList.add("green-dot");
        }

        if (eventList.children.length > 20) eventList.removeChild(eventList.firstChild);
        eventList.appendChild(li);
        eventList.scrollTop = eventList.scrollHeight;
    };

    const inferSource = new EventSource("/inference");
    inferSource.onmessage = function(event) {
        const li = document.createElement("li");
        li.textContent = event.data;
        li.classList.add("green-dot"); // ✅ Always green for inference
        if (inferenceList.children.length > 20) inferenceList.removeChild(inferenceList.firstChild);
        inferenceList.appendChild(li);
        inferenceList.scrollTop = inferenceList.scrollHeight;
    };
</script>
</body>
</html>
    ''')


@app.route('/stream')
def stream():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/events')  # NEW: Event streaming route
def events():
    def event_stream():
        last_len = 0
        while True:
            time.sleep(1)
            if len(event_log) > last_len:
                for msg in event_log[last_len:]:
                    yield f"data: {msg}\n\n"
                last_len = len(event_log)
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")


@app.route('/inference')
def inference():
    def inference_stream():
        last_len = 0
        while True:
            time.sleep(0.5)
            if len(inference_log) > last_len:
                for msg in inference_log[last_len:]:
                    yield f"data: {msg}\n\n"
                last_len = len(inference_log)
    return Response(stream_with_context(inference_stream()), mimetype="text/event-stream")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
