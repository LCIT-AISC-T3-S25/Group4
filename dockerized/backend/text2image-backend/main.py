import os
import io
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image
import torch
import torchvision.transforms as T

from transformers import CLIPTokenizer, CLIPTextModel
from huggingface_hub import HfFolder

from my_model import MyTextToImageModel

# --- HuggingFace Token Handling ---
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    raise RuntimeError("HUGGINGFACE_TOKEN environment variable is not set")
HfFolder.save_token(hf_token)

# --- FastAPI Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model ---
model = MyTextToImageModel().to(device)
model_path = "model/text2image.pth"
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# --- Load Tokenizer & Text Encoder ---
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
text_encoder.eval()

# --- Input Schema ---
class CaptionRequest(BaseModel):
    caption: str

# --- API Endpoint ---
@app.post("/generate-image")
async def generate_image(req: CaptionRequest):
    caption = req.caption.strip()
    if not caption:
        raise HTTPException(status_code=400, detail="Caption cannot be empty")

    inputs = tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
    input_ids = inputs.input_ids.to(device)

    with torch.no_grad():
        text_emb = text_encoder(input_ids).last_hidden_state.mean(dim=1)

    noise = torch.randn(1, 3, 128, 128).to(device)
    timestep = torch.tensor([0.0], device=device)

    with torch.no_grad():
        output = model(noise, timestep, text_emb)

    output_img = output.squeeze(0).cpu()
    output_img = (output_img + 1) / 2
    output_img = output_img.clamp(0, 1)
    pil_img = T.ToPILImage()(output_img)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
