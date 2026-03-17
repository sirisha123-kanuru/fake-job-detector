import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load saved model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens
              if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

def predict_job(title, description, requirements=""):
    combined = title + " " + description + " " + requirements
    cleaned  = clean_text(combined)
    vec      = vectorizer.transform([cleaned])
    pred     = model.predict(vec)[0]
    prob     = model.predict_proba(vec)[0]
    return {
        "label":      "FAKE" if pred == 1 else "REAL",
        "confidence": round(max(prob) * 100, 1),
        "fake_prob":  round(prob[1] * 100, 1),
        "real_prob":  round(prob[0] * 100, 1)
    }

if __name__ == "__main__":
    # Test 1 - obvious fake
    r1 = predict_job(
        title="Work from home - Earn Rs.50,000/day!!",
        description="No experience needed. Send your bank details to start immediately. WhatsApp only."
    )
    print("Test 1 (should be FAKE):", r1)

    # Test 2 - real job
    r2 = predict_job(
        title="Software Engineer",
        description="We are looking for an experienced Python developer to join our team. "
                    "You will work on backend systems, write clean code and participate in code reviews. "
                    "3 years experience required."
    )
    print("Test 2 (should be REAL):", r2)
