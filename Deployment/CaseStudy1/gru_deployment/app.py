from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize Flask app
app = Flask(__name__)

# Load model and tokenizer
model = load_model('/app/gru_sentiment_3class_weighted.keras')
tokenizer = pickle.load(open('/app/tokenizer.pkl', 'rb'))

# Define the inference endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the text from the POST request
        data = request.get_json()
        text = data['text']

        # Tokenize and pad the text input
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=100)  # Adjust maxlen if needed

        # Predict sentiment
        prediction = model.predict(padded_sequence)
        label_map = pickle.load(open('/app/label_map 1.pkl', 'rb'))

        # Get the predicted class
        predicted_class = np.argmax(prediction, axis=1)[0]
        sentiment = label_map[predicted_class]

        return jsonify({'prediction': sentiment})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
