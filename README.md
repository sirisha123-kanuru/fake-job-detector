# Fake Job Posting Detector
> NLP-powered web app that detects scam job postings | SDG 8

## Live Demo
Try it here: [fake-job-detector.streamlit.app](https://fake-job-description-detector.streamlit.app/)

## What it does
- Detects fake job postings with 96.92% accuracy
- Highlights suspicious phrases in red
- Shows confidence score for each prediction
- Supports SDG 8 – Decent Work & Economy

## Tech Stack
- Python, Scikit-learn, NLTK
- TF-IDF Vectorizer + Logistic Regression
- Streamlit (frontend)
- Dataset: 17,880 job postings (Kaggle EMSCAD)

  
## Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | 96.92% |
| Fake Recall | 87% |
| Fake F1-score | 0.73 |


## How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## SDG Impact
Protects job seekers from fraudulent postings — directly
supports UN Sustainable Development Goal 8: Decent Work.
