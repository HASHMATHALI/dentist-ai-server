"""
Dentist AI model server.

Runs the ConvNeXt dental classifier and answers POST /predict in the exact
JSON format the Dentist AI website expects (see src/lib/analysis/service.ts).

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000

The website calls this server over HTTP — it cannot run inside the website
itself. Expose it with a tunnel (cloudflared tunnel --url http://localhost:8000)
or deploy it to a host such as Render, then set the public URL in
VITE_DENTAL_PHOTO_MODEL_URL / VITE_DENTAL_XRAY_MODEL_URL.
"""

import os
import sys
import io
import time
import urllib.request
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "kesar projecct", "model"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "model"))

try:
    from model import create_model
except ImportError:
    create_model = None

# Fallback: build the ConvNeXt-Tiny directly if model.py is not available.
def _build_model(weights_path: str, num_classes: int, device):
    from torchvision.models import convnext_tiny
    import torch.nn as nn

    net = convnext_tiny(weights=None)
    net.classifier[2] = nn.Linear(768, num_classes)
    state_dict = torch.load(weights_path, map_location="cpu")
    # Handle checkpoints that wrap the weights (e.g. {"state_dict": ...}).
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    net.load_state_dict(state_dict, strict=False)
    del state_dict
    return net.to(device)


MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://huggingface.co/hashmath2005/dentist-ai-weights/resolve/main/best_model.pth",
)

app = FastAPI(title="Dentist AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["Abrasion", "Caries", "Crown", "Filling"]

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

EVAL_TRANSFORMS = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

model = None

EXPLANATIONS = {
    "abrasion": "The model detected possible wear on the tooth surface, often from brushing too hard or grinding.",
    "caries": "The model detected a possible area of tooth decay (a cavity) in this image.",
    "crown": "The model detected what appears to be a dental crown (a cap over a tooth).",
    "filling": "The model detected what appears to be an existing dental filling.",
}

RECOMMENDATIONS = {
    "abrasion": [
        "Use a soft-bristled toothbrush and gentle pressure",
        "Mention any grinding or clenching to your dentist",
        "Book a check-up to have the worn area assessed",
    ],
    "caries": [
        "Book a dental visit soon to have this area checked",
        "Brush twice daily with fluoride toothpaste",
        "Reduce sugary snacks and drinks between meals",
    ],
    "crown": [
        "Keep the area clean with regular brushing and flossing",
        "Mention any looseness or sensitivity to your dentist",
    ],
    "filling": [
        "Keep the area clean with regular brushing and flossing",
        "Mention any pain or rough edges to your dentist",
    ],
}


def _find_weights() -> str | None:
    candidates = [
        os.path.join(PROJECT_ROOT, "kesar projecct", "model", "best_model.pth"),
        os.path.join(PROJECT_ROOT, "model", "best_model.pth"),
        os.path.join(HERE, "best_model.pth"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _download_weights(dest: str) -> bool:
    """Stream the weights from MODEL_URL in small chunks (low memory)."""
    try:
        print(f"Downloading model weights from {MODEL_URL} ...")
        with urllib.request.urlopen(MODEL_URL) as resp, open(dest, "wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Weight download failed: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    global model
    weights_path = _find_weights()
    if not weights_path:
        dest = os.path.join(HERE, "best_model.pth")
        if _download_weights(dest):
            weights_path = dest

    if not weights_path:
        print("Warning: best_model.pth not found. Model will not be loaded.")
        return

    print(f"Loading weights from {weights_path}")
    if create_model:
        model = create_model(weights_path=weights_path, num_classes=len(CLASSES), device=device)
    else:
        model = _build_model(weights_path, len(CLASSES), device)
    model.eval()
    print("Model loaded successfully.")


@app.get("/")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


def _result(image_type: str, status: str, supported: bool, findings, explanation, recommendations, started: float):
    return {
        "imageType": image_type,
        "status": status,
        "supported": supported,
        "findings": findings,
        "explanation": explanation,
        "recommendations": recommendations,
        "model": {"id": "dentist-ai-convnext", "version": "1.0", "isMock": False},
        "durationMs": int((time.time() - started) * 1000),
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...), imageType: str = Form("photo")):
    started = time.time()

    if not model:
        return _result(imageType, "error", False, [], "The model is not loaded on the server.", [], started)

    try:
        contents = await image.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = EVAL_TRANSFORMS(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(probs.argmax())

        pred_class = CLASSES[pred_idx]
        key = pred_class.lower()
        confidence = float(probs[pred_idx])

        # This model is a classifier (trained on crops), so it returns one
        # finding for the whole image. The box covers the central region.
        findings: List[Dict[str, Any]] = []
        if confidence >= 0.5:
            findings.append({
                "name": key,
                "confidence": round(confidence, 3),
                "location": "front",
                "bbox": {"x": 0.2, "y": 0.2, "width": 0.6, "height": 0.6},
            })

        if not findings:
            return _result(
                imageType, "no_findings", True, [],
                "The model could not identify a clear finding in this image. This does not rule out a problem — a dentist can examine properly.",
                ["Keep up your regular oral-care routine", "See a dentist for any pain or visible changes"],
                started,
            )

        return _result(
            imageType, "complete", True, findings,
            EXPLANATIONS.get(key, "The model detected a possible finding in this image."),
            RECOMMENDATIONS.get(key, ["Discuss this result with a dentist."]),
            started,
        )
    except Exception as e:
        print(f"Error processing image: {e}")
        return _result(imageType, "error", False, [], "The image could not be processed.", [], started)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
