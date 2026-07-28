import os
import tempfile
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "agri_sight_uploads")
MODEL_PATH = "crop_disease_mobilenet_model.keras"
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
    "Banana Black Sigatoka Disease",
    "Banana Bract Mosaic Virus Disease",
    "Banana Healthy Leaf",
    "Banana Insect Pest Disease",
    "Banana Moko Disease",
    "Banana Panama Disease",
    "Banana Yellow Sigatoka Disease",
    "Black Gram_anthracnose",
    "Black Gram_healthy",
    "Black Gram_leaf crinckle",
    "Black Gram_powdery mildew",
    "Black Gram_yellow mosaic",
    "Broccoli_healthy",
    "Cabbage_healthy",
    "Cardamom_Blight1000",
    "Cardamom_Healthy_1000",
    "Cardamom_Phylosticta_LS_1000",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Eggplant Healthy Leaf",
    "Eggplant Insect Pest Disease",
    "Eggplant Leaf Spot Disease",
    "Eggplant Mosaic Virus Disease",
    "Eggplant Small Leaf Disease",
    "Eggplant White Mold Disease",
    "Eggplant Wilt Disease",
    "Ginger_Bacterial_Wilt",
    "Ginger_Healthy",
    "Jackfruit_Algal_Leaf_Spot",
    "Jackfruit_Black_Spot",
    "Jackfruit_Healthy_Leaf",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Potato___Early_blight",
    "Potato___Late_blight_",
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
    "groundnut_healthy",
    "tomato-healthy"
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
