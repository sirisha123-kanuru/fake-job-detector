import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from preprocess import load_and_prepare

# Load data
df = load_and_prepare()
X = df['clean_text']
y = df['fraudulent']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training on {len(X_train)} samples...")
print(f"Testing on  {len(X_test)} samples...")

# Convert text to numbers using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# Train model
print("\nTraining model...")
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("\nAccuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred,
      target_names=['Real', 'Fake']))

# Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("model.pkl saved!")
print("vectorizer.pkl saved!")
