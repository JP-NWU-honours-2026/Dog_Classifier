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

# Mount static web files
STATIC_DIR = APP_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Mount dataset images for breed image explorer
DATASET_IMAGES_DIR = PROJECT_ROOT / "archive" / "images" / "Images"
if DATASET_IMAGES_DIR.exists():
    app.mount("/archive_images", StaticFiles(directory=str(DATASET_IMAGES_DIR)), name="archive_images")

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
    
    # Check JP_Model first, then local app folder, then Demo_2_breed_model
    candidates = [
        PROJECT_ROOT / "JP_Model" / "best_model.pth",
        APP_DIR / "best_model.pth",
        DEMO_DIR / "best_model.pth",
    ]
    checkpoint_path = None
    for p in candidates:
        if p.exists():
            checkpoint_path = p
            break

    if checkpoint_path is None:
        raise FileNotFoundError(f"Trained checkpoint not found in any candidate locations: {candidates}")

    print(f"[server] Loading model weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    model_name = checkpoint.get("model_name", "efficientnet_v2")
    raw_class_names = checkpoint.get("class_names", {})
    if isinstance(raw_class_names, dict):
        CLASS_NAMES = {int(k): v for k, v in raw_class_names.items()}
    elif isinstance(raw_class_names, list):
        CLASS_NAMES = {i: v for i, v in enumerate(raw_class_names)}
    else:
        CLASS_NAMES = {0: "Chihuahua", 1: "Siberian Husky"}

    MODEL = get_model(model_name=model_name, num_classes=len(CLASS_NAMES), pretrained=False)
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
    MODEL.load_state_dict(state_dict)
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()

    MODEL_METADATA = {
        "model_name": checkpoint.get("model_display_name", getattr(MODEL, "name", model_name)),
        "architecture_family": "EfficientNetV2" if "efficientnet" in model_name.lower() else "ResNet",
        "device": str(DEVICE).upper(),
        "input_resolution": "224 x 224",
        "total_classes": len(CLASS_NAMES),
        "classes": list(CLASS_NAMES.values()),
        "val_accuracy": f"{checkpoint.get('val_acc', 100.0):.1f}%",
        "checkpoint_source": str(checkpoint_path.name),
    }
    print(f"[server] Model ready: {MODEL_METADATA['model_name']} ({len(CLASS_NAMES)} classes) on {DEVICE}")


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


@app.get("/api/samples")
def get_samples() -> list:
    """Returns the curated list of 5 diverse sample dog images for quick testing."""
    return [
        {
            "id": "golden_retriever",
            "name": "Golden Retriever",
            "url": "/static/samples/golden_retriever_sample.jpg",
            "emoji": "🦮",
            "tag": "Sporting Group",
        },
        {
            "id": "german_shepherd",
            "name": "German Shepherd",
            "url": "/static/samples/german_shepherd_sample.jpg",
            "emoji": "🐕‍🦺",
            "tag": "Herding Group",
        },
        {
            "id": "siberian_husky",
            "name": "Siberian Husky",
            "url": "/static/samples/siberian_husky_sample.jpg",
            "emoji": "🐺",
            "tag": "Working Group",
        },
        {
            "id": "french_bulldog",
            "name": "French Bulldog",
            "url": "/static/samples/french_bulldog_sample.jpg",
            "emoji": "🐶",
            "tag": "Companion",
        },
        {
            "id": "chihuahua",
            "name": "Chihuahua",
            "url": "/static/samples/chihuahua_sample.jpg",
            "emoji": "🐕",
            "tag": "Toy Group",
        },
    ]


@app.get("/api/breeds")
def list_breeds() -> list:
    """Returns all 120 dog breeds sorted alphabetically by name."""
    breeds = []
    if DATASET_IMAGES_DIR.exists():
        for folder in DATASET_IMAGES_DIR.iterdir():
            if folder.is_dir():
                parts = folder.name.split("-", 1)
                display_name = parts[1].replace("_", " ").title() if len(parts) > 1 else folder.name
                count = len(list(folder.glob("*.jpg")))
                breeds.append({
                    "id": folder.name,
                    "name": display_name,
                    "folder": folder.name,
                    "image_count": count,
                })
    # Sort strictly alphabetically A-Z by display name
    breeds.sort(key=lambda b: b["name"].lower())
    return breeds


@app.get("/api/breeds/{breed_id}/images")
def get_breed_images(breed_id: str) -> dict:
    """Returns all image URLs for a selected dog breed."""
    folder = DATASET_IMAGES_DIR / breed_id
    if not folder.is_dir():
        # Try finding case-insensitively
        matched = [f for f in DATASET_IMAGES_DIR.iterdir() if f.is_dir() and f.name.lower() == breed_id.lower()]
        if matched:
            folder = matched[0]
        else:
            raise HTTPException(status_code=404, detail=f"Breed folder '{breed_id}' not found")

    parts = folder.name.split("-", 1)
    display_name = parts[1].replace("_", " ").title() if len(parts) > 1 else folder.name
    images = sorted([
        f"/archive_images/{folder.name}/{f.name}"
        for f in folder.glob("*.jpg")
    ])

    return {
        "breed_id": folder.name,
        "name": display_name,
        "total_images": len(images),
        "images": images,
    }


@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Accepts an uploaded dog image, performs preprocessing,
    and returns top-1 predicted breed and top-5 ranked classifications with percentages.
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

    # Extract Top-5 predictions
    sorted_indices = probs.argsort()[::-1]
    top_idx = int(sorted_indices[0])
    top_breed = CLASS_NAMES[top_idx]
    top_confidence = round(float(probs[top_idx]) * 100, 1)

    top5 = []
    for rank, idx in enumerate(sorted_indices[:5], 1):
        top5.append({
            "rank": rank,
            "breed": CLASS_NAMES[int(idx)],
            "probability": round(float(probs[idx]) * 100, 1),
        })

    return {
        "predicted_breed": top_breed,
        "confidence": top_confidence,
        "latency_ms": latency_ms,
        "top5": top5,
        "breakdown": top5,
        "model_name": MODEL_METADATA.get("model_name", "EfficientNetV2-S"),
        "total_classes": len(CLASS_NAMES),
    }


# =============================================================================
# DATASET AUDIT & DUPLICATE REVIEW STUDIO ENDPOINTS
# =============================================================================

from pydantic import BaseModel

JP_DIR = PROJECT_ROOT / "JP_Model"
CANDIDATES_JSON = JP_DIR / "results" / "duplicate_check" / "candidate_duplicates.json"
REMOVALS_JSON = JP_DIR / "results" / "duplicate_check" / "approved_removals.json"


class AuditDecision(BaseModel):
    pair_id: int
    action: str  # "remove_a", "remove_b", "remove_both", "keep_both", "pending"


class BatchDecision(BaseModel):
    action: str  # "apply_recommended", "reset_all"


def _load_candidates() -> list:
    if not CANDIDATES_JSON.exists():
        return []
    import json
    return json.loads(CANDIDATES_JSON.read_text(encoding="utf-8"))


def _save_candidates(candidates: list):
    import json
    CANDIDATES_JSON.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATES_JSON.write_text(json.dumps(candidates, indent=2), encoding="utf-8")


def _sync_approved_removals(candidates: list):
    import json
    removed = []
    for c in candidates:
        status = c.get("status", "")
        if status in ("approved_remove_a", "remove_a"):
            removed.append({
                "key": c["photo_a"]["key"],
                "breed": c["photo_a"]["breed_name"],
                "pile": c["photo_a"]["pile"],
                "pair_id": c["pair_id"],
                "reason": c["reason"]
            })
        elif status in ("approved_remove_b", "remove_b"):
            removed.append({
                "key": c["photo_b"]["key"],
                "breed": c["photo_b"]["breed_name"],
                "pile": c["photo_b"]["pile"],
                "pair_id": c["pair_id"],
                "reason": c["reason"]
            })
        elif status in ("approved_remove_both", "remove_both"):
            removed.append({
                "key": c["photo_a"]["key"],
                "breed": c["photo_a"]["breed_name"],
                "pile": c["photo_a"]["pile"],
                "pair_id": c["pair_id"],
                "reason": c["reason"]
            })
            removed.append({
                "key": c["photo_b"]["key"],
                "breed": c["photo_b"]["breed_name"],
                "pile": c["photo_b"]["pile"],
                "pair_id": c["pair_id"],
                "reason": c["reason"]
            })

    unique_map = {item["key"]: item for item in removed}
    REMOVALS_JSON.write_text(json.dumps(list(unique_map.values()), indent=2), encoding="utf-8")


@app.get("/api/audit/candidates")
def get_audit_candidates():
    """Returns candidate duplicate/conflict pairs for interactive visual review."""
    candidates = _load_candidates()
    return candidates


@app.get("/api/audit/summary")
def get_audit_summary():
    """Returns summary stats for the duplicate review studio."""
    candidates = _load_candidates()
    total = len(candidates)
    pending = sum(1 for c in candidates if c.get("status") == "pending")
    remove_a = sum(1 for c in candidates if c.get("status") in ("approved_remove_a", "remove_a"))
    remove_b = sum(1 for c in candidates if c.get("status") in ("approved_remove_b", "remove_b"))
    remove_both = sum(1 for c in candidates if c.get("status") in ("approved_remove_both", "remove_both"))
    kept = sum(1 for c in candidates if c.get("status") in ("rejected_keep_both", "keep_both"))

    # Count distinct unique photos to remove per split
    import json
    removal_keys = set()
    if REMOVALS_JSON.exists():
        try:
            removals_data = json.loads(REMOVALS_JSON.read_text(encoding="utf-8"))
            for r in removals_data:
                removal_keys.add(r["key"])
        except Exception:
            pass

    return {
        "total_pairs": total,
        "pending_pairs": pending,
        "reviewed_pairs": total - pending,
        "remove_a_count": remove_a,
        "remove_b_count": remove_b,
        "remove_both_count": remove_both,
        "kept_count": kept,
        "unique_photos_to_remove": len(removal_keys)
    }


@app.post("/api/audit/decision")
def post_audit_decision(decision: AuditDecision):
    """Updates the user's decision for a specific candidate duplicate pair."""
    candidates = _load_candidates()
    target = None
    for c in candidates:
        if c.get("pair_id") == decision.pair_id:
            target = c
            break

    if not target:
        raise HTTPException(status_code=404, detail=f"Pair ID {decision.pair_id} not found")

    action = decision.action.lower()
    if action in ("remove_a", "approved_remove_a"):
        target["status"] = "approved_remove_a"
    elif action in ("remove_b", "approved_remove_b"):
        target["status"] = "approved_remove_b"
    elif action in ("remove_both", "approved_remove_both"):
        target["status"] = "approved_remove_both"
    elif action in ("keep_both", "rejected_keep_both", "reject"):
        target["status"] = "rejected_keep_both"
    elif action in ("pending", "reset"):
        target["status"] = "pending"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {decision.action}")

    _save_candidates(candidates)
    _sync_approved_removals(candidates)

    return {"status": "success", "pair_id": decision.pair_id, "new_status": target["status"]}


@app.post("/api/audit/batch")
def post_audit_batch(batch: BatchDecision):
    """Applies batch actions (e.g. apply all recommended decisions, or reset all)."""
    candidates = _load_candidates()
    applied_count = 0

    if batch.action == "apply_recommended":
        for c in candidates:
            if c.get("status") == "pending":
                rec = c.get("default_action", "remove_b")
                c["status"] = f"approved_{rec}"
                applied_count += 1
    elif batch.action == "reset_all":
        for c in candidates:
            c["status"] = "pending"
            applied_count += 1
    else:
        raise HTTPException(status_code=400, detail=f"Unknown batch action: {batch.action}")

    _save_candidates(candidates)
    _sync_approved_removals(candidates)

    return {"status": "success", "action": batch.action, "applied_count": applied_count}


@app.post("/api/audit/generate_clean_manifest")
def generate_clean_manifest_endpoint():
    """Generates split_manifest_clean.json from approved removals."""
    clean_script = JP_DIR / "create_clean_manifest.py"
    if not clean_script.exists():
        raise HTTPException(status_code=500, detail="create_clean_manifest.py not found")

    import subprocess
    result = subprocess.run([sys.executable, str(clean_script)], capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Failed to generate manifest: {result.stderr}")

    summary_file = JP_DIR / "results" / "duplicate_check" / "clean_manifest_summary.json"
    import json
    summary_data = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {}

    return {"status": "success", "summary": summary_data}


if __name__ == "__main__":
    print("\n=======================================================")
    print(" CanineVision AI - Local Web Application")
    print(" Starting server on http://localhost:8000")
    print("=======================================================\n")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False, app_dir=str(APP_DIR))
