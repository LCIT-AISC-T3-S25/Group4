# Group 4 – Deep Learning Model Deployment Assignment

Welcome to our Group 4's academic project on deploying deep learning models using Flask and Docker.
This repository demonstrates basic deployment practices for popular architectures like LSTM, GRU, CNN, and VGG, along with Transfer Learning.

---

## Getting Started

### Run a Deployment Example (LSTM)

```bash
cd lstm_deployment
python app.py
```

Or using Docker:

```bash
docker build -t lstm_app .
docker run -p 5000:5000 lstm_app
```

Then open http://localhost:5000 in your browser.

---

## Continuous Integration

- test-branch-ci.yml: Validates that each deployment folder contains both app.py and Dockerfile.
- code-review.yml: Flags any .ipynb or missing deployment components in pull requests to test.

---

## Sample Outputs

### LSTM Model – Sentiment Analysis with LIME

![LSTM Output]()

### CNN Model – Prediction UI

![CNN Output]()

### VGG Model – Classification Results

![VGG Output]()

### Transfer Learning Deployment – Interface Overview

![Transfer Learning Output]()

---

## Team Members

- Aesha Savani
- Dhruv Boricha
- Aravind Seenivasan
- Taranjot Bindra
- Kabir Pithadiya
- Nidhi Chakarani
- Sai Pranay
- Preet Soni
- Smit Patel
- Nithish Janagam

Thank you for visiting our repository!
