import os
import tempfile
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
import keras

# ======================================================
# 🚨 HOTFIX: PATCH KERAS TO LOAD MODEL SAFELY
# This forces Keras to ignore the 'quantization_config' error
# caused by version mismatches.
# ======================================================
_original_dense_init = keras.layers.Dense.__init__
def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None) # Remove the bad argument
    _original_dense_init(self, *args, **kwargs)
keras.layers.Dense.__init__ = _patched_dense_init

_original_conv2d_init = keras.layers.Conv2D.__init__
def _patched_conv2d_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None) # Remove the bad argument
    _original_conv2d_init(self, *args, **kwargs)
keras.layers.Conv2D.__init__ = _patched_conv2d_init
# ======================================================

# ---------------- CONFIGURATION ----------------
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "agri_sight_uploads")
MODEL_PATH = "crop_disease_cnn_model.keras"
IMAGE_NAME = "test_leaf.jpg"

app = Flask(__name__)
CORS(app)

# Ensure temp folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
print(f"📂 Images will be saved to: {UPLOAD_FOLDER}")

# ---------------- GEMINI AI SETUP ----------------
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
model_gemini = None

if GENAI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GENAI_API_KEY)
        model_gemini = genai.GenerativeModel("gemini-1.5-flash")
        print("✅ Gemini AI configured successfully.")
    except ImportError:
        print("⚠️ 'google-generativeai' library not found. Install it: pip install google-generativeai")
    except Exception as e:
        print(f"⚠️ Gemini configuration error: {e}")
else:
    print("⚠️ WARNING: GEMINI_API_KEY not set. Soil analysis will fail.")

# ---------------- LOAD CNN MODEL ----------------
model = None
try:
    # We don't need custom_objects anymore because we patched the classes globally above
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("✅ CNN model loaded successfully (Applied Hotfix).")
except Exception as e:
    print(f"❌ Critical Error loading CNN model: {e}")

# ---------------- CLASSES ----------------
CLASS_NAMES = [
    # --- Standard PlantVillage Classes (20) ---
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
    "Tomato___healthy",
    
    # --- Extra Classes (Add these to reach 23) ---
    "Banana Black Sigatoka Disease",
    "Banana Bract Mosaic Virus Disease",
    "Banana Healthy Leaf"
]
# ---------------- ROUTES ----------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "Backend is running"}), 200

@app.route("/upload-leaf", methods=["POST"])
def upload_leaf():
    if "image" not in request.files:
        return jsonify({"error": "No image file found"}), 400

    file = request.files["image"]
    save_path = os.path.join(UPLOAD_FOLDER, IMAGE_NAME)
    try:
        file.save(save_path)
        print(f"📸 Image saved to: {save_path}")
        return jsonify({"message": "Image uploaded successfully", "saved_as": IMAGE_NAME}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to save image: {str(e)}"}), 500

@app.route("/predict-leaf", methods=["GET"])
def predict_leaf():
    if model is None:
        return jsonify({"error": "CNN model failed to load. Check server logs."}), 500

    image_path = os.path.join(UPLOAD_FOLDER, IMAGE_NAME)
    if not os.path.exists(image_path):
        return jsonify({"error": "No image uploaded yet"}), 400

    try:
        # Preprocess
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        preds = model.predict(img_array)
        class_index = int(np.argmax(preds))
        confidence = float(np.max(preds))
        
        # Safety check for index
        if class_index < len(CLASS_NAMES):
            result = CLASS_NAMES[class_index]
        else:
            result = "Unknown Disease"

        return jsonify({
            "disease": result,
            "confidence": round(confidence * 100, 2)
        }), 200

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/analyze-soil", methods=["POST"])
def analyze_soil():
    if not model_gemini:
        return jsonify({"suggestion": "AI service unavailable. Check server logs."}), 503

    data = request.json or {}
    # Construct prompt...
    prompt = f"""
    Act as an agricultural expert. Analyze these soil conditions:
    pH: {data.get('ph', 'N/A')}, Moisture: {data.get('moisture', 'N/A')}, 
    N: {data.get('n', 'N/A')}, P: {data.get('p', 'N/A')}, K: {data.get('k', 'N/A')}.
    Provide 3 short, actionable bullet points for soil improvement.
    """
    
    try:
        response = model_gemini.generate_content(prompt)
        return jsonify({"suggestion": response.text}), 200
    except Exception as e:
        return jsonify({"suggestion": "Error generating advice. Ensure API key is valid."}), 500

if __name__ == "__main__":
    app.run(debug=True)