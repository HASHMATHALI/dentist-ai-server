"""
Dentist AI model server (lightweight ONNX version).

Runs the ConvNeXt-Tiny dental classifier (Abrasion / Caries / Crown / Filling)
with onnxruntime instead of PyTorch, so it fits comfortably in a 512 MB host
(Render free tier). Answers POST /predict in the exact JSON contract the
Dentist AI website expects (src/lib/analysis/types.ts).

Run:  uvicorn app:app --host 0.0.0.0 --port $PORT
"""

import gc
import io
import os
import time
import urllib.request

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://huggingface.co/hashmath2005/dentist-ai-weights/resolve/main/dental.onnx",
)
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "dental.onnx")
CLASSES = [c.strip() for c in os.environ.get("CLASSES", "abrasion,caries,crown,filling").split(",")]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MIN_CONFIDENCE = 0.5

EXPLANATIONS = {
    "abrasion": "The model noticed possible wear on the tooth surface, which can come from brushing too hard or grinding.",
    "caries": "The model noticed a possible area of tooth decay (a cavity) in this image.",
    "crown": "The model noticed what appears to be a dental crown (a cap over a tooth).",
    "filling": "The model noticed what appears to be an existing dental filling.",
}
RECOMMENDATIONS = {
    "abrasion": [
        "Use a soft-bristled toothbrush and gentle pressure",
        "Mention any grinding or clenching to your dentist",
        "Consider a check-up to have the worn area assessed",
    ],
    "caries": [
        "Consider a dental visit soon to have this area checked",
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

app = FastAPI(title="Dentist AI model server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

session = None


def _download(url: str, dest: str) -> None:
    print(f"Downloading model from {url} ...", flush=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    os.replace(tmp, dest)
    print("Download complete.", flush=True)


@app.on_event("startup")
def load_model() -> None:
    global session
    try:
        if not os.path.exists(MODEL_PATH):
            _download(MODEL_URL, MODEL_PATH)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])
        gc.collect()
        print("Model loaded successfully.", flush=True)
    except Exception as e:  # keep the server up so / reports the problem
        print(f"Model failed to load: {e}", flush=True)


def _preprocess(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return arr.transpose(2, 0, 1)[None].astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _result(image_type, status, findings, explanation, recommendations, started, supported=True):
    return {
        "imageType": image_type,
        "status": status,
        "supported": supported,
        "findings": findings,
        "explanation": explanation,
        "recommendations": recommendations,
        "model": {"id": "dentist-ai-convnext", "version": "1.1", "isMock": False},
        "durationMs": int((time.time() - started) * 1000),
    }


@app.get("/")
def health():
    return {"status": "ok", "model_loaded": session is not None, "classes": CLASSES}


@app.post("/predict")
async def predict(image: UploadFile = File(...), imageType: str = Form("photo")):
    started = time.time()
    if session is None:
        # 503 makes the website fall back to a clearly labelled sample.
        return JSONResponse({"error": "model not loaded"}, status_code=503)
    try:
        x = _preprocess(await image.read())
    except Exception:
        return _result(imageType, "poor_quality", [], "This image could not be read. Please try a clearer JPG or PNG photo.", [], started, supported=False)

    logits = session.run(None, {"input": x})[0][0]
    probs = _softmax(logits)
    idx = int(probs.argmax())
    key = CLASSES[idx].lower()
    confidence = float(probs[idx])

    if confidence < MIN_CONFIDENCE:
        return _result(
            imageType, "low_confidence", [],
            "The model could not identify a clear finding in this image. This does not rule out a problem — a dentist can examine properly.",
            ["Keep up your regular oral-care routine", "Consider seeing a dentist for any pain or visible changes"],
            started,
        )

    # This model classifies the whole image, so the box marks the central area
    # rather than an exact tooth position.
    finding = {
        "name": key,
        "confidence": round(confidence, 3),
        "location": "upper_front",
        "bbox": {"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5},
    }
    return _result(
        imageType, "complete", [finding],
        EXPLANATIONS.get(key, "The model noticed a possible finding in this image."),
        RECOMMENDATIONS.get(key, ["Consider discussing this result with a dentist."]),
        started,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
