from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import uuid
import os
from PIL import Image
from fastapi.staticfiles import StaticFiles

# Initialize FastAPI app
app = FastAPI()

# Create outputs folder
os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Load ONNX model
onnx_path = "generator.onnx"
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

print("ONNX model loaded. Input name:", input_name)

class GenRequest(BaseModel):
    seed: int = 0

@app.post("/generate")
def generate_image(data: GenRequest):
    # Seed for reproducibility
    np.random.seed(data.seed)

    # Model expects (1, 128, 1, 1)
    z = np.random.randn(1, 128, 1, 1).astype(np.float32)

    # Run inference
    ort_inputs = {input_name: z}
    ort_outs = session.run(None, ort_inputs)
    output = ort_outs[0]

    # Convert tensor to image
    img = np.clip((output[0].transpose(1, 2, 0) + 1) * 127.5, 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img)

    # Save file
    filename = f"out_{uuid.uuid4().hex}.png"
    filepath = os.path.join("outputs", filename)
    img_pil.save(filepath)

    return {"image": filename}
