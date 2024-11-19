import io
import numpy as np
from PIL import Image,ImageDraw
import onnxruntime as ort
from flask import Flask, request, Response,send_file,jsonify
from ultralytics import YOLO
import torch
import cv2
import tempfile
import os

app = Flask(__name__)



# Load the YOLO model (inside the app)
model = YOLO('runs_/detect/train/weights/best.pt')  # or your model path
model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


# Define the directory to save processed videos
OUTPUT_DIR = 'processed_videos'
# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)



print("cuda ------->",torch.cuda.is_available())


@app.route('/')
def index():
    return "Working"



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Determine if the uploaded file is an image or video
        content_type = file.content_type

        if "image" in content_type:
            return process_image(file)
        elif "video" in content_type:
            
            return process_video(file)
        else:
            return jsonify({'error': 'Unsupported file type'})





def process_image(file):
    # Read the image
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Postprocess to get annotated image
    results = model.predict(source=image, save=False)
    annotated_image = results[0].plot()

    # Save the image temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    Image.fromarray(annotated_image).save(temp_file.name)

    # Return the image as a response
    return send_file(temp_file.name, mimetype='image/jpeg')

def process_video(file):
    # Save the uploaded video temporarily
    temp_input_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    file.save(temp_input_video.name)

    # Load video using OpenCV
    cap = cv2.VideoCapture(temp_input_video.name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create an output video file path
    output_video_path = os.path.join(OUTPUT_DIR, f'processed_{os.path.basename(temp_input_video.name)}')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB (YOLO requires RGB format for predictions)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Predict using the YOLO model
        results = model.predict(source=rgb_frame, save=False, conf=0.1)  # Adjust confidence threshold if needed

        # Annotate the frame using YOLO's plotting method
        annotated_frame = results[0].plot()

        # Convert back to BGR format for OpenCV writing
        bgr_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # Write the processed frame to the output video
        out.write(bgr_frame)

    cap.release()
    out.release()

    # Return the path of the processed video
    return jsonify({'message': 'Video processed successfully', 'file_path': output_video_path})

if __name__ == '__main__':
    app.run(debug=True)
