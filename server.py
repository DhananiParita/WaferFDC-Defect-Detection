
import os
import logging

import torch
import torch.nn as nn
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
from torchvision import models, transforms


# --------------------------------------------------
# 1. Flask application setup
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use CPU on Render for compatibility
DEVICE = torch.device("cpu")

MODEL_PATH = os.path.join(BASE_DIR, "wafer_model.pth")
INDEX_PATH = os.path.join(BASE_DIR, "index.html")

CLASS_NAMES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Random",
    "Scratch",
    "Near-full",
    "none",
]

# Match the image preprocessing used during model training.
# If your training pipeline used different settings, update these.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# --------------------------------------------------
# 2. Load the trained model
# --------------------------------------------------

def load_model():
    if not os.path.isfile(MODEL_PATH):
        logger.warning(
            "Model file not found at %s. "
            "The homepage will work, but predictions will be unavailable.",
            MODEL_PATH,
        )
        return None

    try:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

        try:
            checkpoint = torch.load(
                MODEL_PATH,
                map_location=DEVICE,
                weights_only=True,
            )
        except TypeError:
            # Compatibility with older PyTorch versions
            checkpoint = torch.load(
                MODEL_PATH,
                map_location=DEVICE,
            )

        # Support either a plain state_dict or a wrapped checkpoint.
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                checkpoint = checkpoint["model_state_dict"]

        # Remove a possible DataParallel "module." prefix.
        if isinstance(checkpoint, dict):
            checkpoint = {
                key.removeprefix("module."): value
                for key, value in checkpoint.items()
            }

        model.load_state_dict(checkpoint)
        model.to(DEVICE)
        model.eval()

        logger.info("Wafer defect detection model loaded successfully.")
        return model

    except Exception:
        logger.exception("Failed to load the model.")
        return None


model = load_model()


# --------------------------------------------------
# 3. Homepage
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    if not os.path.isfile(INDEX_PATH):
        return jsonify({
            "error": "index.html was not found in the project directory."
        }), 404

    return send_from_directory(BASE_DIR, "index.html")


# --------------------------------------------------
# 4. Health check
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(DEVICE),
    }), 200


# --------------------------------------------------
# 5. Wafer defect prediction
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "error": (
                "The prediction model is unavailable. "
                "Please ensure wafer_model.pth exists and matches "
                "the model architecture and class order."
            )
        }), 503

    if "file" not in request.files:
        return jsonify({
            "error": "No image file was uploaded. Use the field name 'file'."
        }), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({
            "error": "No file was selected."
        }), 400

    try:
        image = Image.open(file.stream).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_index = int(torch.argmax(probabilities).item())

        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(probabilities[predicted_index].item())

        return jsonify({
            "prediction": predicted_class,
            "class": predicted_class,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                CLASS_NAMES[i]: round(float(probabilities[i].item()) * 100, 2)
                for i in range(len(CLASS_NAMES))
            },
        }), 200

    except Exception:
        logger.exception("Prediction failed.")
        return jsonify({
            "error": (
                "Could not process this image. "
                "Please upload a valid wafer-map image."
            )
        }), 400


# --------------------------------------------------
# 6. Start server
# --------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
