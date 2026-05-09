# 🔍 Fake News Detector

A machine learning web app that detects whether a news article is Real or Fake.

## 🛠️ Tech Stack
- Python, Scikit-learn, Pandas
- FastAPI
- HTML, CSS, JavaScript

## 📊 Model Performance
- Algorithm: Random Forest Classifier
- Accuracy: 99.54%

## 🚀 How to Run
1. Install dependencies: `pip install fastapi uvicorn scikit-learn pandas joblib`
2. Start API: `python -m uvicorn app:app --reload`
3. Open `index.html` in browser

## ⚠️ Limitations
- Model trained on political news dataset — performs best on political news articles.

## 🔮 Future Improvements
- Use BERT transformer model for better accuracy across all news categories