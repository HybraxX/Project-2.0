from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
IMAGE_NAME = "test_leaf.jpg"
MODEL_PATH = "crop_disease_cnn_model.keras"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load CNN model (safe & stable)
model = tf.keras.models.load_model(MODEL_PATH)

# ⚠️ UPDATE THIS LIST to match YOUR training dataset order
CLASS_NAMES = [
    "Corn_(maize)___Cercospora_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))   # CNN training size
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "Backend is running"}), 200

@app.route("/upload-leaf", methods=["POST"])
def upload_leaf():
    if "image" not in request.files:
        return jsonify({"error": "No image file found"}), 400

    file = request.files["image"]
    save_path = os.path.join(UPLOAD_FOLDER, IMAGE_NAME)
    file.save(save_path)

    return jsonify({
        "message": "Image uploaded successfully",
        "saved_as": IMAGE_NAME
    }), 200

@app.route("/predict-leaf", methods=["GET"])
def predict_leaf():
    image_path = os.path.join(UPLOAD_FOLDER, IMAGE_NAME)

    if not os.path.exists(image_path):
        return jsonify({"error": "No image uploaded yet"}), 400

    img = preprocess_image(image_path)
    preds = model.predict(img)

    class_index = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return jsonify({
        "disease": CLASS_NAMES[class_index],
        "confidence": round(confidence * 100, 2)
    }), 200

if __name__ == "__main__":
    app.run(debug=True)
