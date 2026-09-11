"""
server.py - FastAPI backend for CanineVision Dog Breed Classifier.

Loads the trained PyTorch checkpoint (best_model.pth) into memory and serves:
1. Static Web UI (HTML, CSS, JavaScript)
2. POST /api/predict - Image classification endpoint
3. GET /api/info - Model metadata endpoint
"""

import io
import sys
import time
from pathlib import Path
from typing import Dict, Any

from PIL import Image
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# Setup paths
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DEMO_DIR = PROJECT_ROOT / "Demo_2_breed_model"

# Ensure local app directory and project paths are in sys.path
for path_dir in [APP_DIR, DEMO_DIR, PROJECT_ROOT]:
    if str(path_dir) not in sys.path:
        sys.path.insert(0, str(path_dir))

# Support both local import and project-relative import
try:
    from models import get_model
except ImportError:
    from Demo_2_breed_model.models import get_model

app = FastAPI(title="CanineVision Dog Classifier", version="1.0.0")

# Mount static files
STATIC_DIR = APP_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global model state
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
CLASS_NAMES = {0: "Chihuahua", 1: "Siberian Husky"}
MODEL_METADATA = {}

# Preprocessing transform
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_classifier():
    """Loads the trained PyTorch model checkpoint into memory."""
    global MODEL, CLASS_NAMES, MODEL_METADATA
    
    # Check local app folder first, then Demo_2_breed_model
    checkpoint_path = APP_DIR / "best_model.pth"
    if not checkpoint_path.exists():
        checkpoint_path = DEMO_DIR / "best_model.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Trained checkpoint not found at: {checkpoint_path}")

    print(f"[server] Loading model weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    model_name = checkpoint.get("model_name", "efficientnet_v2")
    raw_class_names = checkpoint.get("class_names", {0: "Chihuahua", 1: "Siberian Husky"})
    CLASS_NAMES = {int(k): v for k, v in raw_class_names.items()}

    MODEL = get_model(model_name=model_name, num_classes=len(CLASS_NAMES), pretrained=False)
    MODEL.load_state_dict(checkpoint["state_dict"])
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()

    MODEL_METADATA = {
        "model_name": checkpoint.get("model_display_name", getattr(MODEL, "name", model_name)),
        "architecture_family": "EfficientNetV2" if "efficientnet" in model_name.lower() else "ResNet",
        "device": str(DEVICE).upper(),
        "input_resolution": "224 x 224",
        "classes": list(CLASS_NAMES.values()),
        "val_accuracy": f"{checkpoint.get('val_acc', 100.0):.1f}%",
    }
    print(f"[server] Model ready: {MODEL_METADATA['model_name']} on {DEVICE}")


@app.on_event("startup")
def startup_event():
    load_classifier()


@app.get("/")
def serve_index():
    """Serves the main application user interface."""
    index_file = STATIC_DIR / "index.html"
    return FileResponse(str(index_file))


@app.get("/api/info")
def get_info() -> Dict[str, Any]:
    """Returns model metadata and system telemetry."""
    return MODEL_METADATA


@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Accepts an uploaded dog image, performs preprocessing,
    and returns predicted breed with confidence percentages.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image (JPEG/PNG/WEBP).")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # Preprocess image
    tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    # Perform inference and record latency
    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = MODEL(tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    latency_ms = round((time.perf_counter() - start_time) * 1000, 1)

    # Format predictions
    sorted_indices = probs.argsort()[::-1]
    top_idx = int(sorted_indices[0])
    top_breed = CLASS_NAMES[top_idx]
    top_confidence = round(float(probs[top_idx]) * 100, 2)

    breakdown = []
    for idx in sorted_indices:
        breakdown.append({
            "breed": CLASS_NAMES[int(idx)],
            "probability": round(float(probs[idx]) * 100, 2),
        })

    return {
        "predicted_breed": top_breed,
        "confidence": top_confidence,
        "latency_ms": latency_ms,
        "breakdown": breakdown,
        "model_name": MODEL_METADATA.get("model_name", "EfficientNetV2-S"),
    }


if __name__ == "__main__":
    print("\n=======================================================")
    print(" CanineVision AI - Local Web Application")
    print(" Starting server on http://localhost:8000")
    print("=======================================================\n")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False, app_dir=str(APP_DIR))
