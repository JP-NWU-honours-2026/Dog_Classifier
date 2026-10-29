"""
demo_app.py

Serves the baseline CNN through JP's web front end, so the from-scratch model
can be demonstrated live on all 120 breeds.

JP's app was built around the two breed demo (Chihuahua against Siberian husky)
and loads a checkpoint that is not in the repository, since .pth files are not
committed. This script keeps his interface exactly as it is, reads its HTML, CSS
and JavaScript straight from Dog_Classifier_App/static, and answers the same two
endpoints with this model instead:

    GET  /api/info      the model's name, for the status line
    POST /api/predict   an uploaded photograph in, a breed prediction out

Nothing in Dog_Classifier_App is modified or copied.

Usage, from the repository root:

    python Lindani_Model/demo_app.py                  serves results/runs/iter2_deeper
    python Lindani_Model/demo_app.py --run baseline_cnn
    python Lindani_Model/demo_app.py --port 8001

Then open http://localhost:8000

Needs: pip install fastapi uvicorn python-multipart
"""

import argparse
import io
import time

import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from model import BaselineCNN, count_parameters
from split_check import LINDANI_DIR, REPO_ROOT

STATIC_DIR = REPO_ROOT / "Dog_Classifier_App" / "static"
RUNS_DIR = LINDANI_DIR / "results" / "runs"
BREAKDOWN = 5          # how many breeds to show, since 120 bars is unreadable

# Identical to the validation and test preprocessing in dataset.py, so the demo
# sees photographs exactly as the model saw them during evaluation.
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

app = FastAPI(title="Baseline CNN demo", version="1.0.0")
state = {}


def load_model(run_name):
    """Rebuilds whichever architecture the checkpoint was trained with."""
    checkpoint_path = RUNS_DIR / run_name / "best_model.pth"
    if not checkpoint_path.is_file():
        raise SystemExit(
            f"No weights at {checkpoint_path}.\n"
            f"Available runs: {', '.join(p.name for p in RUNS_DIR.iterdir() if p.is_dir())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    class_names = checkpoint["class_names"]

    model = BaselineCNN(
        num_classes=len(class_names),
        dropout=config.get("dropout", 0.3),
        convs_per_block=tuple(config.get("convs_per_block", (1, 1, 1, 1, 1))),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    parameters = count_parameters(model)
    print(f"[demo] {run_name}: {parameters:,} parameters, {len(class_names)} breeds, "
          f"epoch {checkpoint['epoch']}, validation macro F1 "
          f"{checkpoint['val_macro_f1']:.4f}, running on {device}")

    state.update(model=model, device=device, class_names=class_names,
                 name=f"Baseline CNN from scratch ({run_name})", parameters=parameters)


@app.get("/api/info")
def info():
    return {
        "model_name": state["name"],
        "classes": len(state["class_names"]),
        "parameters": state["parameters"],
        "device": str(state["device"]),
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image (JPEG, PNG or WEBP).")

    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Could not read that image: {error}")

    tensor = TRANSFORM(image).unsqueeze(0).to(state["device"])

    started = time.perf_counter()
    with torch.no_grad():
        probabilities = torch.softmax(state["model"](tensor), dim=1)[0].cpu()
    latency_ms = round((time.perf_counter() - started) * 1000, 1)

    # Only the strongest few, because a breakdown of 120 bars is unreadable.
    top = torch.topk(probabilities, k=min(BREAKDOWN, len(state["class_names"])))
    breakdown = [{"breed": state["class_names"][int(i)],
                  "probability": round(100 * float(p), 2)}
                 for p, i in zip(top.values, top.indices)]

    return {
        "predicted_breed": breakdown[0]["breed"],
        "confidence": breakdown[0]["probability"],
        "latency_ms": latency_ms,
        "model_name": state["name"],
        "breakdown": breakdown,
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


def main():
    parser = argparse.ArgumentParser(description="Serve the baseline CNN for a live demo.")
    parser.add_argument("--run", default="iter2_deeper", help="which run's weights to serve")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if not STATIC_DIR.is_dir():
        raise SystemExit(f"The web interface is missing: {STATIC_DIR}")

    load_model(args.run)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    import uvicorn
    print(f"[demo] open http://localhost:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
