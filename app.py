from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


app = Flask(__name__)


# -----------------------------
# Load Model
# -----------------------------

model = load_model("best_model_gru.keras")


# -----------------------------
# Load Tokenizer
# -----------------------------

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


# -----------------------------
# Text Preprocessing
# -----------------------------

def clean_text(text):

    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    # Lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r"[^a-z\s]", "", text)

    # Split into words
    words = text.split()

    # Remove stopwords and apply stemming
    words = [
        ps.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# -----------------------------
# Prediction Function
# -----------------------------

def predict_sentiment(text):

    # Clean text
    cleaned_text = clean_text(text)

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([cleaned_text])

    # Pad sequence
    padded_sequence = pad_sequences(
        sequence,
        maxlen=140,
        padding="post"
    )

    # Predict
    prediction = model.predict(
        padded_sequence,
        verbose=0
    )[0][0]

    # Determine sentiment
    if prediction >= 0.5:
        sentiment = "Positive"
        confidence = prediction * 100
    else:
        sentiment = "Negative"
        confidence = (1 - prediction) * 100

    return sentiment, round(float(confidence), 2)


# -----------------------------
# Home Route
# -----------------------------

@app.route("/")
def home():
    return render_template("sentiment_page.html")


# -----------------------------
# Prediction Route
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict():

    review = request.form.get("review")

    sentiment, confidence = predict_sentiment(review)

    return render_template(
        "sentiment_page.html",
        sentiment=sentiment,
        confidence=confidence,
        review=review
    )


# -----------------------------
# Run Application
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True)