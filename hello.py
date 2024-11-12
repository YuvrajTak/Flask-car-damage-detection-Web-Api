import io
import numpy as np
from PIL import Image
import onnxruntime as rt
from flask import Flask, request, Response,send_file
from ultralytics import YOLO
import torch # Import torch here

app = Flask(__name__)

# Load ONNX model
sess = rt.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])

# Load the YOLO model (inside the app)
model = YOLO('runs/detect/train4/weights/best.pt')  # or your model path
model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))




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
            # Read the image
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            # Postprocess to get annotated image
            # annotated_image = model.predict(source=image, save=False)[0]
            results = model.predict(source=image, save=False)
            annotated_image = results[0].plot()  # Plot the results

            # Save the image temporarily
            temp_file = "temp_image.jpg"  # You can use a different extension
            Image.fromarray(annotated_image).save(temp_file)

            # Return the image as a response
            return send_file(temp_file, mimetype='image/jpeg')



if __name__ == '__main__':
    app.run(debug=True)
