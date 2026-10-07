import io, os, time, gc, urllib.request
import numpy as np, torch, torch.nn as nn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_tiny

# Restrict PyTorch to single-thread to save ~120MB of memory allocation pools
torch.set_num_threads(1)

CLASSES = ["caries", "calculus", "gingivitis", "healthy"]
HEALTHY = {"healthy", "normal"}

MODEL_URL = "https://huggingface.co/hashmath2005/dentist-ai-weights/resolve/main/best_model.pth"
MODEL_PATH = "best_model.pth"

# 1. Stream download in 1MB chunks (prevents RAM buffering)
if not os.path.exists(MODEL_PATH):
    print("Downloading weights from Hugging Face...")
    with urllib.request.urlopen(MODEL_URL) as response, open(MODEL_PATH, "wb") as out_file:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out_file.write(chunk)
    print("Download finished.")

gc.collect()

# 2. Build model architecture
class Net(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.backbone = convnext_tiny(weights=None)
        self.backbone.classifier[2] = nn.Linear(768, n)
        self.target_cam_layer = self.backbone.features[7]
    def forward(self, x):
        return self.backbone(x)

model = Net(len(CLASSES))

# 3. Load weights with immediate garbage collection to release 170MB RAM
with torch.no_grad():
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    del state_dict

# Delete local weights file to preserve container disk
if os.path.exists(MODEL_PATH):
    try:
        os.remove(MODEL_PATH)
    except Exception:
        pass

gc.collect()
model.eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def gradcam_box(x, cls):
    acts, grads = {}, {}
    h1 = model.target_cam_layer.register_forward_hook(lambda m, i, o: acts.update(a=o))
    h2 = model.target_cam_layer.register_full_backward_hook(lambda m, gi, go: grads.update(g=go[0]))
    out = model(x); model.zero_grad(); out[0, cls].backward()
    h1.remove(); h2.remove()
    w = grads["g"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * acts["a"]).sum(1))[0].detach().numpy()
    cam = cam / (cam.max() + 1e-8)
    ys, xs = np.where(cam > 0.5)
    H, W = cam.shape
    if len(xs) == 0: return None
    return {
        "x": float(xs.min() / W),
        "y": float(ys.min() / H),
        "w": float((xs.max() - xs.min() + 1) / W),
        "h": float((ys.max() - ys.min() + 1) / H)
    }

def location_from_box(b):
    if not b: return "upper_front"
    cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
    arch = "upper" if cy < 0.5 else "lower"
    if 0.35 < cx < 0.65: return f"{arch}_front"
    side = "right" if cx < 0.5 else "left"
    return f"{arch}_{side}_{'molar' if cx < 0.2 or cx > 0.8 else 'premolar'}"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["POST"], allow_headers=["*"])

@app.get("/")
def health():
    return {"status": "ok", "model": "convnext-tiny-dental"}

@app.post("/predict")
async def predict(image: UploadFile = File(...), imageType: str = Form("photo")):
    t0 = time.time()
    img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), 1)[0].tolist()
    findings = []
    for i, p in sorted(enumerate(probs), key=lambda t: -t[1]):
        if CLASSES[i] in HEALTHY or p < 0.5: continue
        box = gradcam_box(x.clone().requires_grad_(True), i)
        findings.append({
            "name": CLASSES[i],
            "confidence": round(p, 3),
            "location": location_from_box(box),
            **({"bbox": box} if box else {})
        })
    gc.collect()
    status = "complete" if findings else "no_findings"
    return {
        "imageType": imageType,
        "status": status,
        "supported": True,
        "findings": findings,
        "explanation": "The model highlighted areas that may need clinical evaluation." if findings else "The model did not detect notable dental conditions.",
        "recommendations": ["Consider a professional dental evaluation to confirm these results."],
        "model": {"id": "convnext-tiny-dental", "version": "1.0", "isMock": False},
        "durationMs": int((time.time() - t0) * 1000),
    }
