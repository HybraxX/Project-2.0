# =========================
# Crop Disease CNN Model
# =========================

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten,
    Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# -------------------------
# CONFIG
# -------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
DATASET_PATH = "dataset"   # <-- change if needed
MODEL_NAME = "crop_disease_cnn_model.keras"

# -------------------------
# CHECK DATASET
# -------------------------
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError("Dataset folder not found!")

# -------------------------
# DATA GENERATORS
# -------------------------
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

NUM_CLASSES = train_data.num_classes
CLASS_NAMES = list(train_data.class_indices.keys())

print("Classes Found:")
for i, name in enumerate(CLASS_NAMES):
    print(i, "->", name)

# -------------------------
# CNN MODEL
# -------------------------
model = Sequential([

    Conv2D(32, (3,3), activation='relu',
           input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(NUM_CLASSES, activation='softmax')
])

# -------------------------
# COMPILE MODEL
# -------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------
# TRAIN MODEL
# -------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# -------------------------
# SAVE MODEL
# -------------------------
model.save(MODEL_NAME)
print(f"Model saved as {MODEL_NAME}")

# -------------------------
# SAVE CLASS NAMES
# -------------------------
with open("disease_classes.txt", "w") as f:
    for cls in CLASS_NAMES:
        f.write(cls + "\n")

print("Class labels saved to disease_classes.txt")

# -------------------------
# TEST PREDICTION (OPTIONAL)
# -------------------------
def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions)

    return CLASS_NAMES[predicted_index], confidence

# Example:
# result, conf = predict_disease("test_leaf.jpg")
# print("Prediction:", result)
# print("Confidence:", conf)
