import os
import tempfile
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "agri_sight_uploads")
MODEL_PATH = "crop_disease_mobilenet_plantvillage_model.keras"
IMAGE_NAME = "test_leaf.jpg"
IMG_SIZE = 224

app = Flask(__name__)
CORS(app)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
print(f"📂 Images will be saved to: {UPLOAD_FOLDER}")

# ---------------- LOAD MODEL ----------------
try:
    # safe_mode=False is required to load models that contain Lambda layers
    # (used for MobileNetV2 preprocessing: rescale [0,1] -> [-1,1])
    model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Failed to load model:", e)
    model = None

# ---------------- CLASS NAMES ----------------
CLASS_NAMES = [
    "Corn___Common_Rust",
    "Corn___Gray_Leaf_Spot",
    "Corn___Healthy",
    "Corn___Northern_Leaf_Blight",
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Rice___Brown_Spot",
    "Rice___Healthy",
    "Rice___Leaf_Blast",
    "Rice___Neck_Blast",
    "Sugarcane_Bacterial Blight",
    "Sugarcane_Healthy",
    "Sugarcane_Red Rot",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
    "Wheat___Brown_Rust",
    "Wheat___Healthy",
    "Wheat___Yellow_Rust"
]

# ---------------- ROUTES ----------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "Backend running"}), 200

@app.route("/upload-leaf", methods=["POST"])
def upload_leaf():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    save_path = os.path.join(UPLOAD_FOLDER, IMAGE_NAME)
    file.save(save_path)

    return jsonify({"message": "Image uploaded"}), 200

@app.route("/predict-leaf", methods=["GET"])
def predict_leaf():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    image_path = os.path.join(UPLOAD_FOLDER, IMAGE_NAME)
    if not os.path.exists(image_path):
        return jsonify({"error": "No image uploaded"}), 400

    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    # Normalize to [0, 1] — the Lambda layer inside the model will
    # further rescale to [-1, 1] as MobileNetV2 expects.
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_index = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return jsonify({
        "disease": CLASS_NAMES[class_index],
        "confidence": round(confidence * 100, 2)
    }), 200

if __name__ == "__main__":
    app.run(debug=True)
