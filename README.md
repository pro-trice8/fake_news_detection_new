# 📰 Fake News Detection

An end-to-end **machine learning application for detecting fake and real news articles** using Natural Language Processing (NLP) and multiple classification algorithms.

The project preprocesses news text, converts it into numerical features using **TF-IDF**, compares multiple machine learning models, and provides an interactive **Streamlit application** for real-time predictions.

---

## 🎯 Objective

The goal of this project is to automatically classify a news article as **Fake** or **Real** based on its textual content.

The system applies NLP-based text preprocessing and supervised machine learning to identify patterns that help distinguish misleading news from authentic news.

---

## ⚙️ How It Works

The complete machine learning pipeline follows:

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
Model Selection
     ↓
Streamlit Prediction App
```

1. Text Preprocessing
The news articles are cleaned before being passed to the machine learning models.
The preprocessing pipeline includes:
- Removing unnecessary text patterns using regular expressions
- Cleaning textual noise
- Normalizing the input text
- Preparing article content for vectorization
2. TF-IDF Feature Extraction
The cleaned text is converted into numerical feature vectors using TF-IDF (Term Frequency-Inverse Document Frequency).
TF-IDF assigns higher importance to words that are useful for distinguishing between different types of news while reducing the importance of frequently occurring words.
3. Model Training
Multiple classification algorithms were trained and compared:
- Logistic Regression
- Decision Tree
- Gradient Boosting
- Random Forest
This allows different machine learning approaches to be evaluated using the same text representation.
📊 Model Performance
The evaluated models achieved the following results:
Model	Accuracy	Precision – Fake	Precision – Real
Logistic Regression	92%	89%	94%
Decision Tree	82%	82%	82%
Gradient Boosting	91%	90%	93%
Random Forest	91%	91%	92%


🏆 Best Performing Model
Logistic Regression achieved the highest overall accuracy of 92% among the evaluated models.
Its precision values were:
- Fake News: 89%
- Real News: 94%
🖥️ Streamlit Application
The trained machine learning models are integrated into an interactive Streamlit web application.
The application allows users to:
- Enter or provide a news article
- Select from the available machine learning models
- Predict whether the article is Fake or Real
- View model evaluation information
The application provides an easy way to interact with the trained NLP models without manually running the training pipeline.
🛠️ Tech Stack
Programming & Data Processing
- Python
- Pandas
- NumPy
Natural Language Processing
- TF-IDF Vectorization
- Regular Expressions (Regex)
- Scikit-learn
Machine Learning
- Logistic Regression
- Decision Tree
- Gradient Boosting
- Random Forest
Application
- Streamlit
📁 Project Structure
Fake-News-Detection/
│
├── app.py
├── dataset.csv
├── fake-news-detection.ipynb
├── requirements.txt
└── README.md

File Description
File	Description
app.py	Streamlit application for interactive predictions
dataset.csv	Fake and real news dataset
fake-news-detection.ipynb	Data preprocessing, model training, and evaluation
requirements.txt	Python dependencies
README.md	Project documentation
