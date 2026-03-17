import pandas as pd

df = pd.read_csv("fake_job_postings.csv")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFake vs Real:")
print(df['fraudulent'].value_counts())
print("\nSample fake posting title:")
print(df[df['fraudulent']==1]['title'].iloc[0])