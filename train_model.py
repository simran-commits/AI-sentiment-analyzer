import pandas as pd
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download("stopwords")
nltk.download("wordnet")

# Load dataset
df = pd.read_csv("dataset/sentiment_dataset.csv")

# Keep only required columns
df = df[['review', 'sentiment']]

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)          # Remove HTML tags
    text = re.sub(r'[^a-zA-Z ]', ' ', text)     # Keep only letters
    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

print("Cleaning text...")
df["clean_review"] = df["review"].apply(clean_text)

print("Vectorizing...")
vectorizer = TfidfVectorizer(max_features=10000)

X = vectorizer.fit_transform(df["clean_review"])
y = df["sentiment"]

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training model...")
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", round(accuracy * 100, 2), "%\n")

print(classification_report(y_test, predictions))

joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("✅ Model saved successfully!")