from flask import Flask, render_template, request, jsonify
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

app = Flask(__name__)

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    text = clean_text(data["text"])

    vector = vectorizer.transform([text])

    prediction = model.predict(vector)[0]

    confidence = round(model.predict_proba(vector).max() * 100, 2)

    emoji = "😐"

    if prediction.lower() == "positive":
        emoji = "😍"

    elif prediction.lower() == "negative":
        emoji = "😡"

    return jsonify({
        "prediction": prediction,
        "emoji": emoji,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)