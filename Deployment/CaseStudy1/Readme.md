# Group4: NLP Model Deployment Project

## 📁 Repository Structure

```
Group4/
├── Deployment/
│   └── CaseStudy1/       # Contains deployment code (Flask app, Docker, configs)
├── NLP/
│   └── CaseStudy1/       # Contains model code, notebooks, training scripts
├── .github/
│   └── workflows/        # Contains GitHub Actions (CI policies, checks)
└── README.md             # Project overview and setup guide
```

## 🔀 Branching Policy

- `dev` branch:

  - Contains **only model training and tuning code**
  - No deployment files or Docker scripts allowed

- `test` branch:

  - Contains **deployment and production-ready code**
  - No `.ipynb` or training files allowed

- All PRs to `test` must be **peer reviewed** ✅

## ⚙️ Setup Instructions

### 1. Clone and Navigate

```bash
git clone https://github.com/LCIT-AISC-T3-S25/Group4.git
cd Group4
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r Deployment/CaseStudy1/requirements.txt
```

### 4. Run Model (on `dev` branch)

```bash
cd NLP/CaseStudy1
jupyter notebook
```

### 5. Run Deployment (on `test` branch)

```bash
cd Deployment/CaseStudy1
python app_gru.py  # or app_lstm.py
```

## 🔍 CI/CD & QA

- GitHub Actions validate:
  - Required folders exist
  - Branch policy is enforced
  - Peer reviews are checked before merging
- SonarQube used for static analysis on `dev`
