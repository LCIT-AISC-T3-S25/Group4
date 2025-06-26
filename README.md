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

![LSTM Output](https://github.com/user-attachments/assets/ca1bc91b-782f-4e7b-ab70-e0f87b329481)

### CNN Model – Prediction UI

![CNN Output1](https://github.com/user-attachments/assets/5328a7c6-ce4e-4baa-bba8-963a591131b1)
![CNN Output2](https://github.com/user-attachments/assets/9aece31f-8d7d-4b92-9994-596c90ab114d)

### Transfer Learning Deployment with VGG Model – Classification Results

![VGG Output1](https://github.com/user-attachments/assets/77db77a0-6c48-4174-bdd7-d225405a4830)
![VGG Output2](https://github.com/user-attachments/assets/48271e9f-d304-492e-92d4-2274222f3780)

## SonarQube

![Before](https://github.com/user-attachments/assets/cac6dcf9-f57c-4df1-9c36-b25b6dbda1cd)
![After](https://github.com/user-attachments/assets/ee5b6320-057a-49b4-bc62-a878471eac3f)

## VM (Virtual Machine)

![Output1](https://github.com/user-attachments/assets/a5f380e4-f1f3-4078-add9-cd753576c9bd)
![Output2](https://github.com/user-attachments/assets/83af70ea-40a1-471e-8848-e1c2f12fb2a6)

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
