from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer
import numpy as np
import pickle

app = Flask(__name__)

# Load both models
lstm_model = load_model("models_files/lstm_sentiment_model.h5")
gru_model = load_model("models_files/gru_sentiment_3class_weighted.keras")

# Load tokenizer and label map
with open("models_files/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("models_files/label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

id_to_label = {v: k for k, v in label_map.items()}
class_names = list(label_map.keys())

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = ""
    tweet_text = ""
    lime_html = ""
    selected_model = "lstm"  # Default

    if request.method == "POST":
        tweet_text = request.form.get("tweet")
        selected_model = request.form.get("model")

        if tweet_text:
            sequence = tokenizer.texts_to_sequences([tweet_text])
            padded = pad_sequences(sequence, maxlen=100)

            # Choose model
            model = lstm_model if selected_model == "lstm" else gru_model

            prediction = model.predict(padded)[0]
            neutral_index = label_map['neutral']

            if prediction[neutral_index] >= 0.90:
                sentiment = "Neutral"
            else:
                second_best = np.argsort(prediction)[-2]
                sentiment = id_to_label[second_best].capitalize()

            # LIME Explanation
            explainer = LimeTextExplainer(class_names=class_names)
            explanation = explainer.explain_instance(
                tweet_text,
                lambda x: model.predict(pad_sequences(tokenizer.texts_to_sequences(x), maxlen=100)),
                num_features=6
            )
            lime_html = explanation.as_html()

    return render_template(
        "index.html",
        sentiment=sentiment,
        tweet=tweet_text,
        lime_html=lime_html,
        selected_model=selected_model
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
