from flask import Flask, request, render_template, send_file
import cv2
import torch
import os
import tempfile
import time
from typing import Optional

app = Flask(__name__)
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')  # Load YOLOv5 model

# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/')
def index() -> str:
    """Render the index page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file() -> str:
    """Handle file upload, process the video, and return the processed file.

    Returns:
        str: The response to send to the client.
    """
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    # Save uploaded video
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
    file.save(temp_file.name)

    output_file = process_video(temp_file.name)
    return send_file(output_file, as_attachment=True)

def process_video(video_path: str) -> str:
    """Process the video to detect and annotate cars.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        str: Path to the processed video file.
    """
    cap = cv2.VideoCapture(video_path)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_width = int(original_width * 0.5)
    new_height = int(original_height * 0.5)
    fps = cap.get(cv2.CAP_PROP_FPS)

    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
    out = cv2.VideoWriter(temp_output.name, cv2.VideoWriter_fourcc(*'XVID'), fps, (new_width, new_height))

    car_class_index = 2
    frame_time = 1.0 / fps

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (new_width, new_height))
        results = model(frame_resized)
        detections = results.xyxy[0].cpu().numpy()
        annotated_frame = frame_resized.copy()

        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            if int(cls) == car_class_index:
                cv2.rectangle(
                    annotated_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    annotated_frame,
                    f'Car {conf:.2f}',
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

        out.write(annotated_frame)

        elapsed_time = time.time() - start_time
        if elapsed_time < frame_time:
            time.sleep(frame_time - elapsed_time)

    cap.release()
    out.release()

    return temp_output.name

if __name__ == '__main__':
    app.run(debug=True)
