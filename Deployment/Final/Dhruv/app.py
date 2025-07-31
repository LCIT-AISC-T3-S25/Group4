from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# =====================================================
# Initialize FastAPI
# =====================================================
app = FastAPI(
    title="RAG Chatbot API",
    description="Classifier filters relevance, TinyLlama answers relevant questions",
    version="1.0"
)

# =====================================================
# 1. Classifier model architecture (must match training)
# =====================================================
class RAGModel(nn.Module):
    def __init__(self):
        super(RAGModel, self).__init__()
        self.fc = nn.Linear(768, 2)  # 2 classes

    def forward(self, x):
        return self.fc(x)

# =====================================================
# 2. Load classifier weights (rag_model.pth)
# =====================================================
model = RAGModel()
state_dict = torch.load("rag_model.pth", map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()

# =====================================================
# 3. Load TinyLlama model for chatbot responses
# =====================================================
print("Loading TinyLlama...")
llm_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(llm_name)
llm = AutoModelForCausalLM.from_pretrained(llm_name)

# =====================================================
# 4. Request schema
# =====================================================
class PredictRequest(BaseModel):
    text: str

# =====================================================
# 5. Routes
# =====================================================
@app.get("/")
def home():
    return {"message": "RAG Chatbot API running"}

@app.post("/predict")
def predict(request: PredictRequest):
    question = request.text

    # Step 1: Relevance classification (dummy embedding for now)
    dummy_embedding = torch.randn(1, 768)
    with torch.no_grad():
        output = model(dummy_embedding)
        pred_class = output.argmax(dim=1).item()

    # Step 2: If not relevant, return a polite response
    if pred_class == 0:
        return {
            "input_text": question,
            "answer": "Sorry, I don't have relevant information on that topic."
        }

    # Step 3: Use TinyLlama to generate an answer
    prompt = f"Answer this question in detail:\n{question}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = llm.generate(
        **inputs,
        max_length=200,
        temperature=0.7,
        do_sample=True
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "input_text": question,
        "answer": answer
    }
