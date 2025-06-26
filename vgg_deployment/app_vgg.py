from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64

# --- CONFIG ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ["drink", "food", "inside", "menu", "outside"]

# --- TRANSFORM ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- LOAD VGG16 ---
vgg_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
vgg_model.classifier[6] = nn.Linear(4096, len(class_names))
vgg_model.load_state_dict(torch.load("./vgg.pth", map_location=device))
vgg_model.eval()
vgg_model = vgg_model.to(device)

# --- FASTAPI ---
app = FastAPI()

@app.get("/upload", response_class=HTMLResponse)
def upload_form():
    return """
    <html>
        <head>
            <title>VGG16 Prediction</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f9;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                }
                .container {
                    background-color: white;
                    border-radius: 10px;
                    padding: 30px;
                    width: 60%;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    text-align: center;
                }
                h1 {
                    color: #333;
                    font-size: 32px;
                }
                input[type="file"] {
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    width: 80%;
                    margin-bottom: 20px;
                }
                button {
                    padding: 10px 20px;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 16px;
                }
                button:hover {
                    background-color: #45a049;
                }
                img {
                    margin-top: 20px;
                    max-width: 80%;
                    border-radius: 10px;
                    border: 1px solid #ccc;
                }
                .result {
                    margin-top: 20px;
                    font-size: 22px;
                    color: #333;
                    font-weight: bold;
                }
                .prediction {
                    margin-top: 10px;
                    font-size: 18px;
                    color: #555;
                }
                .progress-bar-container {
                    margin-top: 20px;
                    width: 100%;
                    background-color: #eee;
                    border-radius: 5px;
                    overflow: hidden;
                }
                .progress-bar {
                    height: 20px;
                    text-align: center;
                    color: white;
                    line-height: 20px;
                    font-weight: bold;
                }
                .footer {
                    margin-top: 20px;
                }
                .footer a {
                    text-decoration: none;
                    color: #4CAF50;
                    font-weight: bold;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Upload Image for VGG16 Prediction</h1>
                <form action="/upload" enctype="multipart/form-data" method="post">
                    <input type="file" name="file" accept="image/*" required>
                    <br><br>
                    <button type="submit">Predict</button>
                </form>
            </div>
        </body>
    </html>
    """

@app.post("/upload", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Convert image for display
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    # Transform image
    img_t = transform(img).unsqueeze(0).to(device)

    # VGG16 prediction
    with torch.no_grad():
        vgg_output = vgg_model(img_t)
        vgg_probs = torch.nn.functional.softmax(vgg_output, dim=1)
        vgg_conf, vgg_pred = torch.max(vgg_probs, 1)
        vgg_class = class_names[vgg_pred.item()]
        vgg_conf_pct = vgg_conf.item() * 100

    # Styled HTML result page with prediction and progress bars
    return f"""
    <html>
        <head>
            <title>VGG16 Prediction Result</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f9;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                }}
                .container {{
                    background-color: white;
                    border-radius: 10px;
                    padding: 30px;
                    width: 60%;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    text-align: center;
                }}
                h2 {{
                    color: #333;
                    font-size: 28px;
                }}
                img {{
                    margin-top: 20px;
                    max-width: 80%;
                    border-radius: 10px;
                    border: 1px solid #ccc;
                }}
                .result {{
                    margin-top: 20px;
                    font-size: 22px;
                    color: #333;
                    font-weight: bold;
                }}
                .prediction {{
                    margin-top: 10px;
                    font-size: 18px;
                    color: #555;
                }}
                .progress-bar-container {{
                    margin-top: 20px;
                    width: 100%;
                    background-color: #eee;
                    border-radius: 5px;
                    overflow: hidden;
                }}
                .progress-bar {{
                    height: 20px;
                    text-align: center;
                    color: white;
                    line-height: 20px;
                    font-weight: bold;
                }}
                .footer {{
                    margin-top: 20px;
                }}
                .footer a {{
                    text-decoration: none;
                    color: #4CAF50;
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Prediction: {vgg_class} ({vgg_conf_pct:.2f}%)</h2>
                <img src="data:image/png;base64,{img_base64}" alt="Uploaded Image">
                <div class="result">
                    <div class="prediction">Confidence: {vgg_conf_pct:.2f}%</div>
                </div>
                <div class="progress-bar-container">
                    <div class="progress-bar" style="width: {vgg_conf_pct:.2f}%; background-color: #4CAF50;">{vgg_conf_pct:.2f}%</div>
                </div>
                <div class="footer">
                    <a href="/upload">Upload Another Image</a>
                </div>
            </div>
        </body>
    </html>
    """