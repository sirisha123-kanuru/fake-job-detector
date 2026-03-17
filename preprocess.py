import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens
              if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

def load_and_prepare(filepath="fake_job_postings.csv"):
    df = pd.read_csv(filepath)
    df['combined_text'] = (
        df['title'].fillna('') + ' ' +
        df['company_profile'].fillna('') + ' ' +
        df['description'].fillna('') + ' ' +
        df['requirements'].fillna('')
    )
    df['clean_text'] = df['combined_text'].apply(clean_text)
    df = df[['clean_text', 'fraudulent']].dropna()
    print(f"Dataset ready: {len(df)} rows")
    return df

if __name__ == "__main__":
    df = load_and_prepare()
    print("\nSample cleaned text:")
    print(df['clean_text'].iloc[0][:200])
    print("\nFirst 3 rows:")
    print(df.head(3))
