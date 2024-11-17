from PIL import Image
import io
import os
from io import BytesIO
import time
import cv2
import numpy as np
from tempfile import NamedTemporaryFile
from flask import Flask, request, Response,send_file,jsonify
from ultralytics import YOLO
import torch # Import torch here
from multiprocessing.pool import ThreadPool
import threading
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

# Load ONNX model
# sess = rt.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])

# Load the YOLO model (inside the app)
model = YOLO('runs/detect/train/weights/best.pt')  # or your model path
model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

PROCESSED_FOLDER = "static/processed"
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return "Working"



@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            # Read the file into bytes
            file_bytes = file.read()

            # Check file type (image or video)
            file_type = file.content_type
            if file_type.startswith('image/'):
                return process_image(file_bytes, file.filename)
            elif file_type.startswith('video/'):
                return process_video(file_bytes, file.filename)
            else:
                return jsonify({'error': 'Unsupported file type'})

def process_image(image_bytes, filename):
    try:
        # Convert bytes to PIL image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # YOLO prediction on the image
        results = model.predict(source=image, save=False)
        annotated_image = results[0].plot()  # Annotated image as array

         # Save annotated image to a BytesIO object for response
        image_io = io.BytesIO()
        Image.fromarray(annotated_image).save(image_io, format='JPEG')
        image_io.seek(0)

        # Return the annotated image as a response
        return send_file(
            image_io,
            mimetype='image/jpeg',
            as_attachment=False,
            download_name=f"annotated_{filename}"
        )

    except Exception as e:
        return jsonify({'error': str(e)})

def process_video(video_bytes, filename):
    try:
        # Write the video bytes to a temporary file
        input_video_path = os.path.join(PROCESSED_FOLDER, f"input_{filename}")
        with open(input_video_path, 'wb') as f:
            f.write(video_bytes)

        # Open video using OpenCV
        cap = cv2.VideoCapture(input_video_path)

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Define output path
        output_filename = f"processed_{filename}"
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Process frames
        frame_count = 0
        frame_skip = 1  # Process every 2nd frame for speed
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                out.write(frame)  # Write original frame if skipped
                continue

            # Resize frame for faster inference
            resized_frame = cv2.resize(frame, (640, 640))  # Adjust to YOLO input size
            image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))

            # YOLO prediction
            results = model.predict(source=image, save=False)

            # Annotate the frame
            # Ensure annotations include bounding boxes and labels
            annotated_frame = results[0].plot(labels=True, boxes=True, conf=True)

            # Convert annotated frame back to BGR format for OpenCV
            annotated_frame_bgr = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)
            annotated_frame_resized = cv2.resize(annotated_frame_bgr, (width, height))
            out.write(annotated_frame_resized)

        # Release resources
        cap.release()
        out.release()

        # Return the processed video path
        return jsonify({'message': 'Video processed successfully', 'file_path': output_path})

    except Exception as e:
        return jsonify({'error': str(e)})

    finally:
        # Ensure resources are released even if an error occurs
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'out' in locals():
            out.release()



################################################################################################################



@app.route('/webcam', methods=['GET'])
def webcam_predict():
    try:
        # Start the webcam capture in a separate thread to avoid blocking Flask server
        webcam_thread = threading.Thread(target=webcam_capture_and_preview)
        webcam_thread.start()

        # Define the timer for video saving
        start_time = time.time()

        # Start video capture and saving (this part will also run in Flask's request handler)
        cap = cv2.VideoCapture(0)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create a VideoWriter object to save the video
        output_filename = "webcam_output.mp4"
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Capture frames for 10 seconds
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame for YOLO model (640x640 is commonly used)
            resized_frame = cv2.resize(frame, (640, 640))  
            image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))

            # Predict using YOLO model
            results = model.predict(source=image, save=False)

            # Annotate the frame (labels, boxes, confidence)
            annotated_frame = results[0].plot(labels=True, boxes=True, conf=True)
            annotated_frame_bgr = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)

            # Write the annotated frame to the output video
            out.write(annotated_frame_bgr)

            # Check if 10 seconds have passed
            if time.time() - start_time >= 10:
                break

        # Release webcam and video writer
        cap.release()
        out.release()

        # Check if the video file is created correctly
        if not os.path.exists(output_path):
            return jsonify({'error': 'Video file not created correctly'})

        # Return the video file path in the response
        return jsonify({'message': 'Video processed successfully', 'file_path': output_path})

    except Exception as e:
        return jsonify({'error': str(e)})







def webcam_capture_and_preview():
    # Start video capture from webcam (0 is usually the default webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print('Could not access webcam')
        return

    # Get video properties for output file
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 FPS if FPS is not available
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a window for the webcam preview
    cv2.namedWindow('Webcam Preview', cv2.WINDOW_NORMAL)

    # Capture frames and show the webcam preview
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for YOLO model (640x640 is commonly used)
        resized_frame = cv2.resize(frame, (640, 640))
        image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))

        # Predict using YOLO model
        results = model.predict(source=image, save=False)

        # Annotate the frame (labels, boxes, confidence)
        annotated_frame = results[0].plot(labels=True, boxes=True, conf=True)
        annotated_frame_bgr = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)

        # Show the annotated frame as a preview
        cv2.imshow('Webcam Preview', annotated_frame_bgr)

        # Check for key press (to exit the loop early, press 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release webcam and close the window
    cap.release()
    cv2.destroyAllWindows()








if __name__ == '__main__':
    app.run(debug=True)