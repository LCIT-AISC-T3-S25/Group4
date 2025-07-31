# app/main.py
import json
import pickle
import uvicorn
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from keras.layers import TFSMLayer
from pydantic import BaseModel
from lime.lime_text import LimeTextExplainer

# Load tokenizer
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load label map and invert
with open("model/label_map.json", "r") as f:
    label_map = json.load(f)
inv_label_map = {v: k for k, v in label_map.items()}
id_to_label = {int(v): k for k, v in label_map.items()}

# Load TF model
model_path = "model/sentiment_transformer_savedmodel"
inference_layer = TFSMLayer(model_path, call_endpoint="serving_default")

# Setup FastAPI
app = FastAPI(title="Sentiment Analysis API with LIME")

class TextRequest(BaseModel):
    text: str

def preprocess_text(text):
    max_len = 128
    tokens = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(tokens, maxlen=max_len, padding='post')
    return tf.convert_to_tensor(padded, dtype=tf.float32)

def predict_sentiment(text: str):
    inputs = preprocess_text(text)
    pred_probs_dict = inference_layer(inputs)
    pred_probs = pred_probs_dict['output_0']
    pred_probs = tf.nn.softmax(pred_probs, axis=-1).numpy()

    pred_label_id = int(np.argmax(pred_probs[0]))
    pred_label = id_to_label[pred_label_id]
    confidence = float(pred_probs[0][pred_label_id])

    return pred_label, confidence, pred_probs[0]

# Required for LIME
class_names = [id_to_label[i] for i in range(len(id_to_label))]
explainer = LimeTextExplainer(class_names=class_names)

def lime_predict(texts):
    inputs = tf.convert_to_tensor(
        tf.keras.preprocessing.sequence.pad_sequences(
            tokenizer.texts_to_sequences(texts), maxlen=128, padding='post'
        ), dtype=tf.float32)
    preds = inference_layer(inputs)['output_0']
    return tf.nn.softmax(preds, axis=-1).numpy()

@app.post("/predict")
def predict(req: TextRequest):
    try:
        label, conf, _ = predict_sentiment(req.text)
        return {"label": label, "confidence": conf}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
def explain(req: TextRequest):
    try:
        exp = explainer.explain_instance(req.text, lime_predict, num_features=6)
        explanation = {word: weight for word, weight in exp.as_list()}
        label, conf, _ = predict_sentiment(req.text)
        return {
            "label": label,
            "confidence": conf,
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
