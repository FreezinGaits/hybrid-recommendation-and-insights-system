# 🧠 Hybrid Recommendation & Insights System
## Production-Oriented ML Recommender with Evaluation, Explainability, and Deployment
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-green?logo=streamlit)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue?logo=mlflow)
![Joblib](https://img.shields.io/badge/Joblib-Artifacts-orange?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit-learn-ML-yellow?logo=scikit-learn)
![NetworkX](https://img.shields.io/badge/NetworkX-Graphs-purple?logo=networkx)
![Pandas](https://img.shields.io/badge/Pandas-Data-blue?logo=pandas)
---
## 📌 Overview
This project implements a hybrid product recommendation system that combines multiple recommendation strategies into a single, production-ready pipeline.  
It is designed to demonstrate end-to-end ML engineering skills — from data processing and model design to evaluation, artifact management, and deployment via a web application.  
The system is built to handle sparse, implicit-feedback data, a common real-world challenge in recommendation systems.
---
## 🎯 Problem Statement
Given historical e-commerce order data with implicit feedback (purchases only):  
* Recommend relevant product bundles for a selected product and user  
* Handle cold-start users and products  
* Balance precision, recall, and diversity  
* Provide explainable recommendations  
* Evaluate performance realistically under data sparsity  
---
## 🧩 Solution Approach (Hybrid Recommender)
To address sparsity and cold-start limitations, the system uses a hybrid ensemble of four recommendation strategies:  

1. **Association Rules (Apriori)**  
   * Captures frequent co-purchase patterns  
   * Effective for basket-level insights  

2. **Collaborative Filtering (Item-Based)**  
   * Uses user–item interaction matrix  
   * Computes cosine similarity between products  

3. **Content-Based Filtering**  
   * Uses TF-IDF on product features  
   * Enables recommendations for rarely purchased or new items  

4. **Graph-Based Co-Purchase Analysis**  
   * Builds a weighted co-purchase graph  
   * Captures higher-order relationships between products  

These models are combined into an ensemble recommender to improve robustness over any single method.
---
## 🏗️ System Architecture
```
Raw Data (CSV)
   ↓
Data Cleaning & Feature Engineering
   ↓
Individual Recommender Models
   ├─ Association Rules
   ├─ Collaborative Filtering
   ├─ Content-Based Similarity
   └─ Graph-Based Co-Purchase
   ↓
Ensemble Recommendation Logic
   ↓
Evaluation (Precision@K, Recall@K, MAP)
   ↓
Artifact Packaging (joblib)
   ↓
Streamlit App (Inference & Visualization)
```
---
## 📊 Evaluation Methodology

### Metrics Used
* Precision@K  
* Recall@K  
* Mean Average Precision (MAP)  

### Evaluation Setup
* Evaluated on held-out user interactions  
* Implicit feedback setting (purchases only)  
* No explicit negatives available  
* Metrics computed per user and aggregated  
---
## 📈 Results (Current Dataset)
```
Average Precision (MAP): 0.1952
Precision@3: 0.1444
Recall@3: 0.1339
Precision@5: 0.1267
Recall@5: 0.1861
```
---
## 🔍 Interpretation of Results
These metric values are expected and realistic given:  
* High sparsity (most users have ≤2 purchases)  
* Cold-start users and products  
* Implicit feedback without negative samples  
* High product diversity and noisy feature semantics  

In real-world recommender systems, recall, coverage, and diversity are often prioritized over raw precision in early or sparse data regimes.
---
## 🧪 Experiment Tracking
* Metrics and artifacts are logged using MLflow  
* Enables reproducibility and comparison across runs  
* Tracked artifacts include:  
  * Trained recommender components  
  * Encoders  
  * Final deployment bundle  
---
## 📦 Artifacts
The training pipeline produces a single deployment artifact:  
```
recommender_artifacts.joblib
```

**Contents:**  
* products_df  
* label_encoders  
* bundle_rules  
* item_similarity_df  
* content_similarity_df  
* G (co-purchase graph)  
* user_item_matrix  
* tfidf_vectorizer  

This artifact is directly consumed by the application layer.
---
## 🖥️ Application (Deployment)
A Streamlit web app is provided to demonstrate real-time inference and analytics:  

### Features
* Product & user selection  
* Individual model recommendations  
* Ensemble recommendations  
* Price compatibility & diversity filtering  
* Temporal insights (recent co-purchases, trends)  
* Interactive visualizations (graphs, charts, tables)  

### Run locally:
```bash
streamlit run app.py
```
---
## 🧠 Key Design Decisions
* Hybrid approach chosen to mitigate cold-start and sparsity issues  
* Item-based CF preferred over user-based due to limited user history  
* Content-based similarity added for explainability and new products  
* Graph modeling used to capture higher-order co-purchase structure  
* Artifact-based deployment for clean separation of training and inference  
---
## 🚀 Future Improvements
* Temporal-aware recommendation models  
* Better feature normalization and semantic consistency  
* Negative sampling for improved evaluation  
* Online learning / incremental updates  
* A/B testing framework for live feedback  
* Integration with real user intent signals  
---
## 🛠️ Tech Stack
* Python  
* Pandas, NumPy  
* Scikit-learn  
* MLxtend  
* NetworkX  
* MLflow  
* Streamlit  
* Joblib  
---
## 👨‍💻 Author

Gautam Sharma  
Computer Science Engineering (B.Tech)  
Focused on Machine Learning, MLOps, and Production AI Systems  

This project was built to reflect real-world ML engineering practices, not just model training.
---
⭐ This repository demonstrates how machine learning systems are **built, evaluated, deployed, and maintained** in real production environments.
