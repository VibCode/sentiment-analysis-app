# Sentiment Analysis Web App

A simple web app that analyzes the **sentiment of product reviews** (Positive or Negative) using **Natural Language Processing (NLP)** and a **Naive Bayes classifier**.

This project was built from scratch in Python and deployed locally using Flask.

Libraries: pandas, nltk, scikit-learn, flask, joblib
---

## Features 

-  Input a review through a user-friendly web interface
-  Get real-time sentiment prediction (Positive or Negative)
-  Simple and lightweight app — great for beginners
-  Trained using a publicly available movie review dataset from NLTK

---

## Tech Stack

- **Frontend**: HTML, Bootstrap (via Flask templates)
- **Backend**: Python, Flask
- **NLP**: NLTK (Natural Language Toolkit)
- **ML Model**: Naive Bayes Classifier
- **Packaging**: `joblib`
- **Version Control**: Git + GitHub

---

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Train the model:
   python train_model.py

3. Start the web app:
   python app.py

4. Visit in your browser:
   http://127.0.0.1:5000/

## Future Improvements

1. Deploy on a cloud platform like Render, Vercel, or Heroku
2. Train on real product reviews (e.g., Amazon)
3. Improve model accuracy with advanced NLP (e.g., TF-IDF, LSTM)

