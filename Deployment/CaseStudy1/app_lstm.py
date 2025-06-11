from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# config
MAX_LEN = 100
CLASSES = ['negative','neutral','positive']

# load
app = Flask(__name__)
lstm = load_model('models/lstm_sentiment_model.h5')
with open('tokenizer.pickle','rb') as f:
    tok = pickle.load(f)

def preprocess(text):
    seq = tok.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=MAX_LEN)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get('text','')
    x = preprocess(text)
    scores = lstm.predict(x)[0].tolist()
    label = CLASSES[scores.index(max(scores))]
    return jsonify({'label': label, 'scores': scores})

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000)