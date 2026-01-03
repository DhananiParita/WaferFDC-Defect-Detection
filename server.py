import os
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)
CORS(app)

# --- 1. HARDCODED CLASS ORDER (CRITICAL) ---
# This MUST match the alphabetical order of folders used during training.
# Do not use os.listdir() here, as it can be random on different computers.
CLASS_NAMES = [
    'Center', 
    'Donut', 
    'Edge-Loc', 
    'Edge-Ring', 
    'Loc', 
    'Near-full', 
    'None', 
    'Random', 
    'Scratch'
]

# --- 2. DEFECT KNOWLEDGE BASE ---
DEFECT_INFO = {
    "Center": {"cause": "CVD gas flow obstruction or spin-coating error."},
    "Donut": {"cause": "Thermal annealing unevenness or cooling issue."},
    "Edge-Loc": {"cause": "Wafer handling gripper or edge bead removal failure."},
    "Edge-Ring": {"cause": "Etching plasma uniformity or edge seal breach."},
    "Loc": {"cause": "Localized contamination, droplets, or particles."},
    "Near-full": {"cause": "Catastrophic process failure or machine stop."},
    "Random": {"cause": "Cleanroom air filtration failure or dust."},
    "Scratch": {"cause": "Robotic handling damage or CMP polishing error."},
    "None": {"cause": "Nominal yield. Process is stable."}
}

# --- 3. MODEL SETUP ---
device = torch.device("cpu") # Use CPU for server stability
MODEL_PATH = 'wafer_model.pth'

# Initialize Model Architecture
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

# Load Weights
try:
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print(f"✅ Model Loaded! Expecting {len(CLASS_NAMES)} classes.")
        print(f"📋 Classes: {CLASS_NAMES}")
    else:
        print("❌ CRITICAL ERROR: 'wafer_model.pth' not found in folder!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Image Preprocessing (Must match training exactly)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    try:
        # Convert to RGB to ensure 3 channels (even if input is Black/White)
        image = Image.open(file.stream).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            conf, pred_idx = torch.max(probs, 0)
        
        pred_class = CLASS_NAMES[pred_idx]
        conf_score = conf.item() * 100
        
        # --- DEBUG PRINT (Look at your terminal!) ---
        print(f"🔍 Analyzing Image... Predicted: {pred_class} ({conf_score:.2f}%)")
        
        # Prepare graph data
        graph_data = {name: float(probs[i]) * 100 for i, name in enumerate(CLASS_NAMES)}
        
        info = DEFECT_INFO.get(pred_class, {"cause": "Unknown defect type."})
        
        return jsonify({
            'class': pred_class,
            'confidence': f"{conf_score:.1f}",
            'cause': info['cause'],
            'graph_data': graph_data
        })
        
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Server starting on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000)