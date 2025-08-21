# Phase 3: Machine Learning with Lahman Baseball Database

This repository is part of my **10-Project Data & AI Governance Roadmap**, demonstrating how governance, trust, and compliance come together across the AI lifecycle.  
Phase 3 focuses on **building predictive models, applying explainability techniques, and analyzing bias** to ensure trust and transparency in AI/ML systems.

---

## Project Overview

### Project 5: Predictive ML Model – Player & Team Performance

Using the Lahman Baseball Database (1871–2023), I built predictive models to forecast player and team performance:

- **Objective:** Predict player batting averages, home runs, or team wins based on historical data  
- **Features:** Player age, position, historical stats, team context, salary, career trends  
- **Models:** Baseline regression models, XGBoost / LightGBM for improved performance  
- **Evaluation:** Train/test splits, cross-validation, RMSE, accuracy metrics  

**Output:** Jupyter notebook with model training, evaluation, and feature importance analysis

---

### Project 6: Explainability & Bias Analysis – Baseball Stats

To ensure trust and fairness in ML models:

- Applied **SHAP** and **LIME** to interpret model predictions  
- Analyzed potential biases such as:  
  - Era-based performance bias  
  - Position-based performance differences  
  - Salary or team-based disparities  
- Documented findings in charts and explanatory notes  

**Output:** Notebook + charts + blog post interpreting model explanations and bias analysis

---

## Technologies Used

- **Python:** `pandas`, `numpy` for data handling  
- **Machine Learning:** `scikit-learn`, `xgboost`, `lightgbm` for modeling  
- **Explainability:** `shap`, `lime` for interpretability  
- **Cloud:** AWS SageMaker for training and experimentation  
- **Visualization:** Matplotlib, Seaborn, Plotly for feature importance and bias charts  

**Alternative Tools / Platforms:**

- TensorFlow / PyTorch for deep learning approaches  
- Azure ML / GCP Vertex AI for cloud ML pipelines  
- Fairlearn / InterpretML for additional fairness and bias analysis  

---

## Strategic Alignment with Governance Roadmap

**Data Foundations Matter**  
- Built predictive models on clean, queryable datasets generated in Phases 1–2  
- Demonstrated how historical and structured data supports AI/ML workflows  

**Lineage & Observability**  
- Documented feature engineering steps and data sources  
- Logged model training parameters and evaluation metrics for reproducibility  

**Security & Privacy by Design**  
- Handled sensitive player data (birthdates, minor league identifiers) safely  
- Demonstrated patterns applicable to healthcare, financial, or other regulated datasets  

**Trust as a Product Feature**  
- Used explainable AI techniques to **make model predictions interpretable**  
- Identified and visualized potential bias for transparent AI systems  

**Enterprise + Public Perspective**  
- **AWS-native path:** SageMaker for training, S3 for datasets, CloudWatch for logging  
- **Local / public path:** Python notebooks + Jupyter for sharing models and insights

