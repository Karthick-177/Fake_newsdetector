import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

true = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")

true["label"] = 0
fake["label"] = 1

true["content"] = true["title"] + " " + true["text"]
fake["content"] = fake["title"] + " " + fake["text"]

df = pd.concat([true[["content", "label"]], fake[["content", "label"]]], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(df["content"], df["label"], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ model.pkl and vectorizer.pkl saved!")