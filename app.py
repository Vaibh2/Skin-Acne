from flask import Flask, Response, render_template, redirect, url_for, request, send_file
import torch
import cv2
import numpy as np
from PIL import Image
import os
import io
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLOv5 model using torch.hub
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp15/weights/last.pt', force_reload=True)

# Video capture
cap = cv2.VideoCapture(0)  # Change to 0 if 1 doesn’t work; it refers to the webcam index

def generate_frames():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Perform acne detection
        results = model(frame)
        processed_frame = np.squeeze(results.render())  # Render model results on the frame

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()

        # Yield frame in an HTTP-compliant format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Define a route for the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Define route for main page (index page)
@app.route('/')
def index():
    return render_template('index.html')

# Define route for video page
@app.route('/show_video')
def show_video():
    return render_template('video_feed.html')

# Define route for uploading an image
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file part", 400

        file = request.files['image']
        if file.filename == '':
            return "No selected file", 400

        # Read image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = np.array(img)

        # Perform acne detection
        results = model(img)
        processed_img = np.squeeze(results.render())

        # Convert result image to byte format
        _, buffer = cv2.imencode('.jpg', processed_img)
        response_img = buffer.tobytes()

        # Return image as HTTP response
        return Response(response_img, mimetype='image/jpeg')

    return render_template('upload.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
