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

# --- LOAD CNN ---
cnn_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
cnn_model.fc = nn.Linear(cnn_model.fc.in_features, len(class_names))
cnn_model.load_state_dict(torch.load("./cnn.pth", map_location=device))
cnn_model.eval()
cnn_model = cnn_model.to(device)

# --- FASTAPI ---
app = FastAPI()

@app.get("/upload", response_class=HTMLResponse)
def upload_form():
    return """
    <html>
        <head>
            <title>CNN Image Prediction</title>
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
                    font-size: 20px;
                    color: #333;
                    font-weight: bold;
                }
                .prediction {
                    margin-top: 10px;
                    font-size: 18px;
                    color: #555;
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
                <h1>Upload Image for CNN Prediction</h1>
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

    # CNN prediction
    with torch.no_grad():
        cnn_output = cnn_model(img_t)
        cnn_probs = torch.nn.functional.softmax(cnn_output, dim=1)
        cnn_conf, cnn_pred = torch.max(cnn_probs, 1)
        cnn_class = class_names[cnn_pred.item()]
        cnn_conf_pct = cnn_conf.item() * 100

    # Return a stylish HTML result page with prediction
    return f"""
    <html>
        <head>
            <title>CNN Prediction Result</title>
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
                <h2>Prediction: {cnn_class} ({cnn_conf_pct:.2f}%)</h2>
                <img src="data:image/png;base64,{img_base64}" alt="Uploaded Image">
                <div class="result">
                    <div class="prediction">Confidence: {cnn_conf_pct:.2f}%</div>
                </div>
                <div class="footer">
                    <a href="/upload">Upload Another Image</a>
                </div>
            </div>
        </body>
    </html>
    """