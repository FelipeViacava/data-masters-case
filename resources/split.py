import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/raw.csv")

train, test = train_test_split(df, test_size=0.25, random_state=42, stratify=df["TARGET"])

train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)

print("Train and test sets saved to data/ folder")