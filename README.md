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

![CNN Output1](https://github.com/user-attachments/assets/cfa18d4e-3558-40de-94a5-6e3f65394185)
![CNN Output2](https://github.com/user-attachments/assets/8d57a501-e39b-45f2-9c5c-37c0677ee6e2)

### Transfer Learning Deployment with VGG Model – Classification Results

![VGG Output1](https://github.com/user-attachments/assets/60f972ff-bb82-4bdf-b0f4-8d41bc60f378)
![VGG Output2](https://github.com/user-attachments/assets/b02fc75c-42c7-412a-a8e7-ecaf8222e1fd)

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
