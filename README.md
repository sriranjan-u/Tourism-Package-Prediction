---
title: Tourism Package Prediction
emoji: ✈️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

#Tourism Package Prediction

An end-to-end MLOps project designed to predict customer purchase behavior for a new Tourism Package. This project automates the entire machine learning lifecycle—from data ingestion and preprocessing to model training and cloud deployment.

## Project Links
* **GitHub Repository**: [https://github.com/sriranjan-u/Tourism-Package-Prediction]
* **Hugging Face Space (Demo)**: [https://huggingface.co/spaces/Sriranjan/Tourism-Package-Pred]
* **Dataset & Model Hub**: [https://huggingface.co/Sriranjan/Tourism-Package-Pred]


The project utilizes a modular CI/CD architecture managed via GitHub Actions:

1.  **Data Layer**: Raw and processed data splits (`Xtrain.csv`, `Xtest.csv`) are versioned and hosted at `Sriranjan/Tourism-Package-Pred`.
2.  **Experimentation**: Model training is tracked using **MLflow**, capturing hyperparameters and classification metrics (Accuracy, F1-Score).
3.  **Automation**: The `.github/workflows/mlops.yml` pipeline triggers on every push to:
    * Clean and encode raw data.
    * Train the XGBoost Classifier.
    * Synchronize deployment assets with the Hugging Face Space.
4.  **Deployment**: The prediction service is containerized via **Dockerfile** and served via **Streamlit** on Hugging Face Spaces.

## Directory Structure

.
├── .github/workflows/
│   └── mlops.yml             # CI/CD Automation Pipeline
├── app/
│   ├── app.py                # Streamlit Web Application
│   ├── requirements.txt      # Production Dependencies
│   └── Dockerfile            # Container Specification
├── data/
│   ├── raw/                  # Original Dataset (tourism.csv)
│   └── processed/            # Encoded Train/Test Splits
├── models/
│   └── tourism_xgb_model.joblib
├── src/
│   ├── preprocess.py         # Data Cleaning & Encoding Logic
│   ├── train.py              # Training Script & MLflow Logging
│   └── evaluate.py           # Model Validation & Metrics
└── README.md
