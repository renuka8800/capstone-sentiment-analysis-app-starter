import pickle
from flask import Flask, render_template, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()

model = None
tokenizer = None


def load_keras_model():
    global model
    model = load_model("models/uci_sentimentanalysis.h5")


def load_tokenizer():
    global tokenizer
    with open("models/tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)


@app.before_request
def before_first_request():
    # Load once per worker to avoid repeated loads / memory issues
    global model, tokenizer
    if model is None:
        load_keras_model()
    if tokenizer is None:
        load_tokenizer()


def sentiment_analysis(text: str) -> float:
    user_sequences = tokenizer.texts_to_sequences([text])
    user_sequences_matrix = sequence.pad_sequences(user_sequences, maxlen=1225)
    prediction = model.predict(user_sequences_matrix, verbose=0)
    return round(float(prediction[0][0]), 2)


@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    user_text = ""

    if request.method == "POST":
        user_text = request.form.get("user_text", "")
        sentiment = analyzer.polarity_scores(user_text)  # VADER
        sentiment["custom model positive"] = sentiment_analysis(user_text)  # Keras

    return render_template("form.html", sentiment=sentiment, user_text=user_text)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
