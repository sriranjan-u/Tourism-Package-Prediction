---
title: Sriranjan's Tourism Package Prediction
emoji: ✈️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Sriranjan's Tourism Package Prediction

An end-to-end MLOps project engineered by Sriranjan Uppoor. This system automates the machine learning lifecycle—from data ingestion and XGBoost model training to a containerized cloud deployment on Hugging Face Spaces.

## Project Links
* **GitHub Repository**: [https://github.com/sriranjan-u/Tourism-Package-Prediction]
* **Live Demo**: [https://huggingface.co/spaces/Sriranjan/Tourism-Package-Pred]

## MLOps Pipeline
The project utilizes a modular CI/CD architecture managed via GitHub Actions:

1.  **Data Ingestion**: Raw data is versioned and processed into training/testing splits.
2.  **Model Training**: An XGBoost Classifier is trained with results tracked via **MLflow**.
3.  **Deployment**: Assets are synchronized to a Docker-based Hugging Face Space using a customized GitHub Actions workflow.

## Directory Structure
```text
.
├── .github/workflows/mlops.yml  # Automated CI/CD Pipeline
├── app/
│   ├── app.py                 # Streamlit UI
│   ├── requirements.txt       # App Dependencies (XGBoost, etc.)
│   └── Dockerfile             # Container Config
├── src/                       # Source Code (Train/Eval/Prep)
└── README.md
