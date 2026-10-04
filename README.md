# 📰 Fake News Detection

An end-to-end **machine learning application for detecting fake and real news articles** using natural language processing (NLP) and multiple classification algorithms.

The project preprocesses news text, converts it into numerical features using **TF-IDF**, and compares multiple machine learning models to identify the most effective classifier. The final models are integrated into an interactive **Streamlit application** for real-time predictions.

---

## 🎯 Objective

The goal of this project is to automatically classify a news article as **Fake** or **Real** based on its textual content.

The system applies NLP-based text preprocessing and machine learning classification to identify patterns associated with misleading or authentic news.

---

## ⚙️ How It Works

The complete pipeline follows:

```text
News Dataset
     ↓
Text Cleaning & Preprocessing
     ↓
Regex-based Text Normalization
     ↓
TF-IDF Feature Extraction
     ↓
Train/Test Split
     ↓
Multiple ML Classifiers
     ↓
Model Evaluation
     ↓
Best Model Selection
     ↓
Streamlit Prediction App
