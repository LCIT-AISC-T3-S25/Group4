from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
import json
from model import CausalTransformer, encode
from lime.lime_text import LimeTextExplainer

# Load updated stoi
with open("stoi.json", "r") as f:
    stoi = json.load(f)
itos = {v: k for k, v in stoi.items()}

# Initialize Flask app
app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint
ckpt = torch.load("tuned_qa_transformer_model.pth", map_location=device)

# Create model with current vocab size
vocab_size = len(stoi)  # e.g., 8766
model = CausalTransformer(vocab_size, d_model=256, nhead=8, num_layers=4)

# Load only compatible weights
def load_partial_weights(model, ckpt):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in ckpt.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"✅ Loaded {len(pretrained_dict)} matching layers from checkpoint.")

load_partial_weights(model, ckpt)
model.to(device).eval()

# LIME wrapper class
class QAWrapper:
    def __init__(self, model, stoi, device):
        self.model = model
        self.stoi = stoi
        self.device = device

    def predict(self, texts):
        vocab_size = self.model.embed.num_embeddings
        input_ids = []
        for text in texts:
            ids = encode(text, self.stoi)
            ids = [i if i < vocab_size else 0 for i in ids]  # clamp invalid indices
            input_ids.append(ids)

        input_tensor = torch.tensor(input_ids).to(self.device)
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        return probs

lime_explainer = LimeTextExplainer(class_names=list(stoi.keys()))
lime_wrapper = QAWrapper(model, stoi, device)

@app.route("/")
def index():
    return render_template("templates\index.html")

@app.route("/answer", methods=["POST"])
def answer():
    data = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please enter a question."})

    tokens = encode(question, stoi)
    unknown = [w for w in question.lower().split() if w not in stoi]
    if unknown:
        return jsonify({"answer": f"Unknown word(s): {', '.join(unknown)}. Try rephrasing."})

    input_ids = torch.tensor([tokens]).to(device)
    with torch.no_grad():
        output = model(input_ids)
        pred_tokens = output.argmax(-1).squeeze().tolist()
        decoded = [itos[t] for t in pred_tokens if t != 0]
    return jsonify({"answer": " ".join(decoded)})

@app.route("/explain", methods=["POST"])
def explain():
    data = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"explanation": []})

    exp = lime_explainer.explain_instance(
        question,
        lime_wrapper.predict,
        num_features=10,
        labels=[0]
    )
    explanation = exp.as_list(label=0)
    return jsonify({"explanation": explanation})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6000)