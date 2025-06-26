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

![LSTM Output](https://github.com/user-attachments/assets/ff50ed11-254e-4f7b-95de-01ad73de5c73)

### CNN Model – Prediction UI

![CNN Output1](https://github.com/user-attachments/assets/266e843e-6497-4c8e-8dce-3841eecc650f)
![CNN Output2](https://github.com/user-attachments/assets/b5e2692a-fa7f-4510-a8d6-de7080a9b163)

### Transfer Learning Deployment with VGG Model – Classification Results

![VGG Output1](https://github.com/user-attachments/assets/90850df2-3c6c-4364-81f6-b973fac47913)
![VGG Output2](https://github.com/user-attachments/assets/f507eeeb-e75d-41c4-9f88-08c89689bca2)

## SonarQube

![Before](https://github.com/user-attachments/assets/e20aa9db-106c-48aa-b396-2196e82fdfb7)
![After](https://github.com/user-attachments/assets/57de29d5-3a83-4ea5-9c36-ca3e5653fe1c)

## VM (Virtual Machine)

![Output1](https://github.com/user-attachments/assets/146c7a88-b9aa-42e5-91a0-e230d9ef56f1)
![Output2](https://github.com/user-attachments/assets/aed457de-1bc3-4aac-a0a2-c387b8bb83c3)

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
