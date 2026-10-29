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
from typing import Dict, Any, List, Optional
import torch.nn as nn
from pydantic import BaseModel

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
CURRENT_MODEL_ID = "jp_efficientnet_simplified"
MODEL = None
CLASS_NAMES = {}
MODEL_METADATA = {}
MODEL_CACHE = {}

# Preprocessing transform
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Catalog of the 3 per-person models on the simplified split (114 breeds)
SIMPLIFIED_MODELS_CATALOG = [
    {
        "id": "jp_efficientnet_simplified",
        "author": "JP",
        "model_display_name": "JP: EfficientNetV2-S (Simplified Split)",
        "short_name": "EfficientNetV2-S",
        "architecture_family": "EfficientNetV2",
        "parameters": "20.3M",
        "classes_count": 114,
        "split_manifest": "split_manifest_simplified.json",
        "val_accuracy": "85.8%",
        "test_accuracy": "84.2%",
        "description": "JP's EfficientNetV2-S transfer model fine-tuned on the 114-breed simplified split.",
        "checkpoint_path": PROJECT_ROOT / "JP_Model" / "results" / "runs" / "simplified_transfer" / "best_model.pth",
        "badge": "EfficientNetV2 ? 114 Breeds",
    },
    {
        "id": "lindani_mobilenet_simplified",
        "author": "Lindani",
        "model_display_name": "Lindani: MobileNetV3-Large (Simplified Split)",
        "short_name": "MobileNetV3-Large",
        "architecture_family": "MobileNetV3",
        "parameters": "4.3M",
        "classes_count": 114,
        "split_manifest": "split_manifest_simplified.json",
        "val_accuracy": "70.4%",
        "test_accuracy": "68.7%",
        "description": "Lindani's MobileNetV3-Large matching baseline CNN capacity (4.3M params) on 114 breeds.",
        "checkpoint_path": PROJECT_ROOT / "Lindani_Model" / "mobilenet" / "results" / "runs" / "mobilenet_simplified" / "best_model.pth",
        "badge": "MobileNetV3 ? 114 Breeds",
    },
    {
        "id": "sulaiman_swin_simplified",
        "author": "Sulaiman",
        "model_display_name": "Sulaiman: Swin-T Transformer (Simplified Split)",
        "short_name": "Swin-T Transformer",
        "architecture_family": "Vision Transformer (Swin)",
        "parameters": "27.6M",
        "classes_count": 114,
        "split_manifest": "split_manifest_simplified.json",
        "val_accuracy": "Pending Training",
        "test_accuracy": "Pending Training",
        "description": "Sulaiman's Swin Transformer Tiny architecture on 114 breeds (weights not yet trained).",
        "checkpoint_path": PROJECT_ROOT / "Sulaiman_Model" / "results" / "runs" / "swin_t_simplified" / "best_model.pth",
        "badge": "Swin-T ? Not Yet Trained",
    },
]


def load_classifier(model_id: Optional[str] = None):
    """Loads a simplified split model checkpoint into memory by ID."""
    global MODEL, CLASS_NAMES, MODEL_METADATA, CURRENT_MODEL_ID, MODEL_CACHE

    target_id = model_id or CURRENT_MODEL_ID or "jp_efficientnet_simplified"

    if target_id in MODEL_CACHE:
        cached = MODEL_CACHE[target_id]
        MODEL = cached["model"]
        CLASS_NAMES = cached["classes"]
        MODEL_METADATA = cached["metadata"]
        CURRENT_MODEL_ID = target_id
        return MODEL_METADATA

    catalog_map = {m["id"]: m for m in SIMPLIFIED_MODELS_CATALOG}
    if target_id not in catalog_map:
        target_id = "jp_efficientnet_simplified"

    entry = catalog_map[target_id]
    checkpoint_path = entry["checkpoint_path"]

    if not checkpoint_path.exists():
        if target_id == "sulaiman_swin_simplified":
            raise HTTPException(
                status_code=400,
                detail="Sulaiman's Swin-T model has not been trained on the simplified split yet."
            )
        raise FileNotFoundError(f"Trained checkpoint not found at: {checkpoint_path}")

    print(f"[server] Loading model weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    raw_class_names = checkpoint.get("class_names", {})
    if isinstance(raw_class_names, dict):
        class_names = {int(k): v for k, v in raw_class_names.items()}
    elif isinstance(raw_class_names, list):
        class_names = {i: v for i, v in enumerate(raw_class_names)}
    else:
        class_names = {i: f"Class {i}" for i in range(entry["classes_count"])}

    num_classes = len(class_names)
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
    disp_name = checkpoint.get("model_display_name", "")

    if "mobilenet" in target_id or "mobilenet" in disp_name.lower() or "classifier.3.weight" in state_dict:
        from torchvision.models import mobilenet_v3_large
        model = mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[2] = nn.Dropout(p=0.3, inplace=True)
        model.classifier[3] = nn.Linear(in_features, num_classes)
        model.name = "Transfer-MobileNetV3-Large"
    elif "swin" in target_id:
        from torchvision.models import swin_t
        model = swin_t(weights=None)
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
        model.name = "Transfer-Swin-T"
    else:
        from torchvision.models import efficientnet_v2_s
        model = efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
        model.name = "Transfer-EfficientNetV2-S"

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_k = k[6:] if k.startswith("model.") else k
        cleaned_state_dict[new_k] = v

    model.load_state_dict(cleaned_state_dict)
    model = model.to(DEVICE)
    model.eval()

    val_acc_val = checkpoint.get("val_acc")
    val_acc_str = f"{val_acc_val:.1f}%" if val_acc_val else entry["val_accuracy"]

    metadata = {
        "model_id": target_id,
        "author": entry["author"],
        "model_name": entry["model_display_name"],
        "model_display_name": getattr(model, "name", entry["model_display_name"]),
        "architecture_family": entry["architecture_family"],
        "parameters": entry["parameters"],
        "device": str(DEVICE).upper(),
        "input_resolution": "224 x 224",
        "total_classes": num_classes,
        "classes": list(class_names.values()),
        "val_accuracy": val_acc_str,
        "test_accuracy": entry["test_accuracy"],
        "checkpoint_source": str(checkpoint_path.name),
        "description": entry["description"],
        "badge": entry["badge"],
    }

    MODEL_CACHE[target_id] = {
        "model": model,
        "classes": class_names,
        "metadata": metadata,
    }

    MODEL = model
    CLASS_NAMES = class_names
    MODEL_METADATA = metadata
    CURRENT_MODEL_ID = target_id
    print(f"[server] Model ready: {metadata['model_name']} ({num_classes} classes) on {DEVICE}")
    return metadata


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


class SelectModelRequest(BaseModel):
    model_id: str


@app.get("/api/models")
def get_models():
    """Lists the 3 per-person models on the simplified split."""
    return {
        "active_model_id": CURRENT_MODEL_ID,
        "models": [
            {
                "id": m["id"],
                "author": m["author"],
                "name": m["model_display_name"],
                "short_name": m["short_name"],
                "architecture": m["architecture_family"],
                "parameters": m["parameters"],
                "classes_count": m["classes_count"],
                "val_accuracy": m["val_accuracy"],
                "test_accuracy": m["test_accuracy"],
                "description": m["description"],
                "badge": m["badge"],
                "available": m["checkpoint_path"].exists(),
                "is_active": (m["id"] == CURRENT_MODEL_ID),
            }
            for m in SIMPLIFIED_MODELS_CATALOG
        ],
    }


@app.post("/api/models/select")
def select_model_endpoint(payload: SelectModelRequest):
    """Switches the active classification model."""
    try:
        meta = load_classifier(payload.model_id)
        return {
            "status": "success",
            "active_model_id": CURRENT_MODEL_ID,
            "metadata": meta,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


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
async def predict_image(file: UploadFile = File(...), model_id: Optional[str] = None):
    """
    Accepts an uploaded dog image, performs preprocessing,
    and returns top-1 predicted breed and top-5 ranked classifications with percentages.
    """
    global MODEL, CLASS_NAMES, MODEL_METADATA, CURRENT_MODEL_ID
    if model_id and model_id != CURRENT_MODEL_ID:
        load_classifier(model_id)

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
        "model_id": CURRENT_MODEL_ID,
        "author": MODEL_METADATA.get("author", "JP"),
        "model_name": MODEL_METADATA.get("model_name", "EfficientNetV2-S"),
        "architecture": MODEL_METADATA.get("architecture_family", "EfficientNetV2"),
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


# =============================================================================
# BREED SIMPLIFICATION & MERGER STUDIO ENDPOINTS
# =============================================================================

from typing import List, Optional

SIMPLIFIED_MANIFEST_PATH = JP_DIR / "split_manifest_simplified.json"
MERGER_SUMMARY_PATH = JP_DIR / "results" / "merger_summary.json"

SUGGESTED_CLUSTERS = [
    {
        "id": "staffordshire",
        "category": "Terriers",
        "default_name": "Staffordshire Terrier",
        "description": "Staffordshire Bull Terrier & American Staffordshire Terrier",
        "breeds": ["Staffordshire Bullterrier", "American Staffordshire Terrier"],
    },
    {
        "id": "foxhound",
        "category": "Hounds",
        "default_name": "Foxhound",
        "description": "English Foxhound & Walker Hound",
        "breeds": ["English Foxhound", "Walker Hound"],
    },
    {
        "id": "poodle",
        "category": "Toy & Companion",
        "default_name": "Poodle",
        "description": "Toy, Miniature, and Standard Poodles",
        "breeds": ["Toy Poodle", "Miniature Poodle", "Standard Poodle"],
    },
    {
        "id": "schnauzer",
        "category": "Pinscher & Schnauzer",
        "default_name": "Schnauzer",
        "description": "Miniature, Standard, and Giant Schnauzers",
        "breeds": ["Miniature Schnauzer", "Standard Schnauzer", "Giant Schnauzer"],
    },
    {
        "id": "welsh_corgi",
        "category": "Herding & Pastoral",
        "default_name": "Welsh Corgi",
        "description": "Pembroke & Cardigan Welsh Corgis",
        "breeds": ["Pembroke", "Cardigan"],
    },
    {
        "id": "springer_spaniel",
        "category": "Gundogs & Spaniels",
        "default_name": "Springer Spaniel",
        "description": "English & Welsh Springer Spaniels",
        "breeds": ["English Springer", "Welsh Springer Spaniel"],
    },
    {
        "id": "setter",
        "category": "Gundogs & Setters",
        "default_name": "Setter",
        "description": "English, Irish, and Gordon Setters",
        "breeds": ["English Setter", "Irish Setter", "Gordon Setter"],
    },
    {
        "id": "swiss_mountain_dog",
        "category": "Working & Mountain",
        "default_name": "Swiss Mountain Dog",
        "description": "Bernese, Greater Swiss, Appenzeller, and Entlebucher Sennenhunde",
        "breeds": ["Bernese Mountain Dog", "Greater Swiss Mountain Dog", "Appenzeller", "Entlebucher"],
    },
    {
        "id": "coonhound",
        "category": "Hounds & Hunting",
        "default_name": "Coonhound",
        "description": "Black-and-Tan Coonhound, Bluetick, and Redbone",
        "breeds": ["Black-And-Tan Coonhound", "Bluetick", "Redbone"],
    },
    {
        "id": "belgian_shepherd",
        "category": "Herding & Sheepdogs",
        "default_name": "Belgian Shepherd",
        "description": "Groenendael and Malinois Belgian Sheepdogs",
        "breeds": ["Groenendael", "Malinois"],
    },
    {
        "id": "collie_shepherd",
        "category": "Herding & Sheepdogs",
        "default_name": "Collie & Sheepdog",
        "description": "Collie, Border Collie, and Shetland Sheepdog",
        "breeds": ["Collie", "Border Collie", "Shetland Sheepdog"],
    },
    {
        "id": "norfolk_norwich",
        "category": "Terriers",
        "default_name": "Norfolk & Norwich Terrier",
        "description": "Norfolk Terrier and Norwich Terrier",
        "breeds": ["Norfolk Terrier", "Norwich Terrier"],
    },
    {
        "id": "bulldog",
        "category": "Molosser & Companion",
        "default_name": "Bulldog",
        "description": "Boston Bull and French Bulldog",
        "breeds": ["Boston Bull", "French Bulldog"],
    },
    {
        "id": "retriever",
        "category": "Gundogs & Retrievers",
        "default_name": "Retriever",
        "description": "Golden, Labrador, Flat-Coated, Curly-Coated, and Chesapeake Bay Retrievers",
        "breeds": ["Golden Retriever", "Labrador Retriever", "Flat-Coated Retriever", "Curly-Coated Retriever", "Chesapeake Bay Retriever"],
    },
]


class MergeGroupItem(BaseModel):
    id: str
    target_name: str
    breeds: List[str]


class MergerPayload(BaseModel):
    source_manifest: str = "split_manifest_clean.json"
    groups: List[MergeGroupItem]


def _get_breed_thumbnail_and_folder(display_name: str) -> tuple[str, str, int]:
    """Finds folder name, representative thumbnail URL, and total image count for a breed."""
    if not DATASET_IMAGES_DIR.exists():
        return "", "", 0

    target_sub = display_name.lower().replace(" ", "_").replace("-", "_")
    matched_folder = None

    for folder in DATASET_IMAGES_DIR.iterdir():
        if not folder.is_dir():
            continue
        parts = folder.name.split("-", 1)
        folder_sub = parts[1].lower() if len(parts) > 1 else folder.name.lower()
        if folder_sub == target_sub or folder.name.lower().endswith(target_sub):
            matched_folder = folder
            break

    if not matched_folder:
        for folder in DATASET_IMAGES_DIR.iterdir():
            if folder.is_dir() and target_sub in folder.name.lower():
                matched_folder = folder
                break

    if not matched_folder:
        return "", "", 0

    images = list(matched_folder.glob("*.jpg"))
    thumb_url = f"/archive_images/{matched_folder.name}/{images[0].name}" if images else ""
    return matched_folder.name, thumb_url, len(images)


@app.get("/api/merger/source_manifests")
def list_source_manifests():
    """Lists available manifests in JP_Model that can be used as the base for simplification."""
    import json
    manifest_files = [
        {
            "id": "split_manifest_clean.json",
            "name": "split_manifest_clean.json (Cleaned Dataset)",
            "description": "Recommended. Cleaned split with duplicate pairs and cross-split conflicts removed.",
            "is_recommended": True,
        },
        {
            "id": "split_manifest.json",
            "name": "split_manifest.json (Original Raw Split)",
            "description": "Original raw 70/15/15 Stanford Dogs dataset split.",
            "is_recommended": False,
        },
    ]

    results = []
    for m in manifest_files:
        path = JP_DIR / m["id"]
        exists = path.exists()
        info = {**m, "exists": exists, "total_classes": 0, "counts": {}}
        if exists:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                info["total_classes"] = len(data.get("classes", {}))
                info["counts"] = data.get("metadata", {}).get("counts", {})
            except Exception as e:
                info["error"] = str(e)
        results.append(info)

    simplified_exists = SIMPLIFIED_MANIFEST_PATH.exists()
    simplified_info = None
    if simplified_exists:
        try:
            sim_data = json.loads(SIMPLIFIED_MANIFEST_PATH.read_text(encoding="utf-8"))
            simplified_info = {
                "total_classes": len(sim_data.get("classes", {})),
                "counts": sim_data.get("metadata", {}).get("counts", {}),
                "source_manifest": sim_data.get("metadata", {}).get("source_manifest", ""),
                "merged_groups_count": sim_data.get("metadata", {}).get("merged_groups_count", 0),
            }
        except Exception:
            pass

    return {
        "manifests": results,
        "simplified_manifest_exists": simplified_exists,
        "simplified_manifest_info": simplified_info,
    }


@app.get("/api/merger/suggested_clusters")
def get_suggested_clusters():
    """Returns pre-curated similarity clusters enriched with thumbnail previews and live image counts."""
    enriched = []
    for c in SUGGESTED_CLUSTERS:
        breeds_detail = []
        total_cluster_images = 0
        for b in c["breeds"]:
            folder, thumb, count = _get_breed_thumbnail_and_folder(b)
            total_cluster_images += count
            breeds_detail.append({
                "name": b,
                "folder": folder,
                "thumbnail": thumb,
                "image_count": count,
            })

        enriched.append({
            "id": c["id"],
            "category": c["category"],
            "default_name": c["default_name"],
            "description": c["description"],
            "total_images": total_cluster_images,
            "breeds": breeds_detail,
        })

    return enriched


def _compute_merge_simulation(source_manifest_name: str, groups: List[MergeGroupItem]) -> dict:
    import json
    src_path = JP_DIR / source_manifest_name
    if not src_path.exists():
        raise HTTPException(status_code=404, detail=f"Source manifest '{source_manifest_name}' not found")

    manifest_data = json.loads(src_path.read_text(encoding="utf-8"))
    raw_classes = manifest_data.get("classes", {})

    # Build mapping from original display_name to (target_class_name, group_id, is_merged)
    breed_to_target = {}
    valid_groups = []
    for g in groups:
        if len(g.breeds) < 2:
            continue
        target_name = g.target_name.strip()
        if not target_name:
            continue
        valid_groups.append(g)
        for b in g.breeds:
            breed_to_target[b] = {
                "target_name": target_name,
                "group_id": g.id,
                "is_merged": True,
                "constituent_breeds": g.breeds,
            }

    # Identify all final class names
    original_names = [c["display_name"] for c in raw_classes.values()]
    all_final_names = set()
    for orig in original_names:
        if orig in breed_to_target:
            all_final_names.add(breed_to_target[orig]["target_name"])
        else:
            all_final_names.add(orig)

    sorted_final_names = sorted(list(all_final_names), key=lambda x: x.lower())
    target_name_to_id = {name: idx for idx, name in enumerate(sorted_final_names)}

    # Build detailed class list
    class_details = {}
    for name in sorted_final_names:
        class_details[name] = {
            "new_label": target_name_to_id[name],
            "display_name": name,
            "is_merged": False,
            "constituent_breeds": [name],
            "folders": [],
            "split_counts": {"train": 0, "val": 0, "test": 0, "total": 0},
            "thumbnails": [],
        }

    # Enrich constituent breeds and folders
    for k, c in raw_classes.items():
        orig_name = c["display_name"]
        folder = c.get("folder", "")
        _, thumb, _ = _get_breed_thumbnail_and_folder(orig_name)

        if orig_name in breed_to_target:
            target_name = breed_to_target[orig_name]["target_name"]
            class_details[target_name]["is_merged"] = True
            class_details[target_name]["constituent_breeds"] = breed_to_target[orig_name]["constituent_breeds"]
            if folder and folder not in class_details[target_name]["folders"]:
                class_details[target_name]["folders"].append(folder)
            if thumb and thumb not in class_details[target_name]["thumbnails"]:
                class_details[target_name]["thumbnails"].append(thumb)
        else:
            class_details[orig_name]["constituent_breeds"] = [orig_name]
            if folder:
                class_details[orig_name]["folders"] = [folder]
            if thumb:
                class_details[orig_name]["thumbnails"] = [thumb]

    # Calculate exact counts across splits
    splits_data = manifest_data.get("splits", {})
    for split_key in ("train", "val", "test"):
        for item in splits_data.get(split_key, []):
            orig_breed = item.get("breed_name", "")
            target_name = breed_to_target[orig_breed]["target_name"] if orig_breed in breed_to_target else orig_breed
            if target_name in class_details:
                class_details[target_name]["split_counts"][split_key] += 1
                class_details[target_name]["split_counts"]["total"] += 1

    orig_total_classes = len(raw_classes)
    new_total_classes = len(sorted_final_names)
    reduction = orig_total_classes - new_total_classes
    reduction_pct = round((reduction / orig_total_classes) * 100, 1) if orig_total_classes > 0 else 0.0

    return {
        "source_manifest": source_manifest_name,
        "original_classes_count": orig_total_classes,
        "resulting_classes_count": new_total_classes,
        "classes_reduced": reduction,
        "reduction_percentage": reduction_pct,
        "active_merged_groups_count": len(valid_groups),
        "classes": list(class_details.values()),
        "target_name_to_id": target_name_to_id,
        "breed_to_target": {k: v["target_name"] for k, v in breed_to_target.items()},
    }


@app.post("/api/merger/preview")
def preview_merge(payload: MergerPayload):
    """Calculates merge statistics and preview of resulting simplified classes."""
    sim = _compute_merge_simulation(payload.source_manifest, payload.groups)
    sim.pop("target_name_to_id", None)
    sim.pop("breed_to_target", None)
    return sim


@app.post("/api/merger/generate")
def generate_simplified_manifest(payload: MergerPayload):
    """
    Applies the class merge mapping and generates JP_Model/split_manifest_simplified.json
    with newly indexed continuous labels (0 to K-1).
    """
    import json
    from datetime import datetime

    sim = _compute_merge_simulation(payload.source_manifest, payload.groups)
    target_name_to_id = sim["target_name_to_id"]
    breed_to_target = sim["breed_to_target"]

    src_path = JP_DIR / payload.source_manifest
    raw_manifest = json.loads(src_path.read_text(encoding="utf-8"))

    simplified_classes = {}
    for cls_item in sim["classes"]:
        new_id_str = str(cls_item["new_label"])
        simplified_classes[new_id_str] = {
            "display_name": cls_item["display_name"],
            "is_merged": cls_item["is_merged"],
            "constituent_breeds": cls_item["constituent_breeds"],
            "folders": cls_item["folders"],
        }

    simplified_splits = {"train": [], "val": [], "test": []}
    split_counts = {"train": 0, "val": 0, "test": 0, "total": 0}

    for pile in ("train", "val", "test"):
        for item in raw_manifest.get("splits", {}).get(pile, []):
            orig_breed = item.get("breed_name", "")
            target_name = breed_to_target.get(orig_breed, orig_breed)
            new_label = target_name_to_id[target_name]

            transformed_item = {
                "path": item["path"],
                "label": new_label,
                "breed_folder": item["breed_folder"],
                "breed_name": target_name,
            }
            if "original_breed_name" not in transformed_item and orig_breed != target_name:
                transformed_item["original_breed_name"] = orig_breed

            simplified_splits[pile].append(transformed_item)
            split_counts[pile] += 1
            split_counts["total"] += 1

    simplified_manifest = {
        "metadata": {
            **raw_manifest.get("metadata", {}),
            "manifest_type": "simplified_merged",
            "source_manifest": payload.source_manifest,
            "total_classes": len(simplified_classes),
            "original_classes": sim["original_classes_count"],
            "merged_groups_count": sim["active_merged_groups_count"],
            "classes_reduced": sim["classes_reduced"],
            "reduction_percentage": f"{sim['reduction_percentage']}%",
            "counts": split_counts,
            "generated_at": datetime.now().isoformat(),
        },
        "classes": simplified_classes,
        "splits": simplified_splits,
    }

    SIMPLIFIED_MANIFEST_PATH.write_text(
        json.dumps(simplified_manifest, indent=2), encoding="utf-8"
    )

    summary = {
        "generated_at": datetime.now().isoformat(),
        "source_manifest": payload.source_manifest,
        "output_manifest": "split_manifest_simplified.json",
        "original_classes": sim["original_classes_count"],
        "resulting_classes": len(simplified_classes),
        "classes_reduced": sim["classes_reduced"],
        "reduction_percentage": sim["reduction_percentage"],
        "merged_groups_applied": [
            {"group_id": g.id, "target_name": g.target_name, "breeds": g.breeds}
            for g in payload.groups
            if len(g.breeds) >= 2 and g.target_name.strip()
        ],
        "split_counts": split_counts,
    }
    MERGER_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MERGER_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "status": "success",
        "output_file": str(SIMPLIFIED_MANIFEST_PATH.name),
        "summary": summary,
        "resulting_classes_count": len(simplified_classes),
    }


@app.get("/api/merger/download")
def download_simplified_manifest():
    """Serves split_manifest_simplified.json for browser download."""
    if not SIMPLIFIED_MANIFEST_PATH.exists():
        raise HTTPException(status_code=404, detail="split_manifest_simplified.json has not been generated yet")
    return FileResponse(
        str(SIMPLIFIED_MANIFEST_PATH),
        filename="split_manifest_simplified.json",
        media_type="application/json",
    )


if __name__ == "__main__":
    print("\n=======================================================")
    print(" CanineVision AI - Local Web Application")
    print(" Starting server on http://localhost:8000")
    print("=======================================================\n")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False, app_dir=str(APP_DIR))
