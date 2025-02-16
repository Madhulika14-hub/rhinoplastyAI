from flask import Flask, request, jsonify, send_file
import torch
import cv2
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import io
import os

app = Flask(__name__)

# Load Stable Diffusion Inpainting Model
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
).to("cuda")

# Function to Create Nose Mask
def create_nose_mask(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    mask = np.zeros_like(gray)
    for (x, y, w, h) in faces:
        nose_x, nose_y, nose_w, nose_h = int(x + w * 0.4), int(y + h * 0.5), int(w * 0.2), int(h * 0.2)
        mask[nose_y:nose_y + nose_h, nose_x:nose_x + nose_w] = 255

    mask = cv2.GaussianBlur(mask, (15, 15), 10)
    return Image.fromarray(mask)

# Function to Modify Nose
def modify_nose(image, nose_style):
    styles = {
        "Slim": "a smaller, symmetrical nose with a refined bridge",
        "Sharp": "a sharp, well-defined nose with a chiseled bridge",
        "Natural": "a soft, natural-looking nose with subtle contours",
        "Upturned": "a slightly upturned nose with a delicate shape",
    }

    mask = create_nose_mask(image)
    result = pipe(prompt=styles[nose_style], image=image, mask_image=mask).images[0]
    
    output = io.BytesIO()
    result.save(output, format="PNG")
    output.seek(0)
    return output

@app.route('/process', methods=['POST'])
def process_image():
    file = request.files['image']
    nose_style = request.form.get('nose_style', 'Natural')

    image = Image.open(file).convert("RGB")
    modified_image = modify_nose(image, nose_style)

    return send_file(modified_image, mimetype='image/png')

@app.route('/')
def home():
    return send_file('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
