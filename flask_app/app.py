from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import dagshub

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# ========================== CONFIGURATION ==========================
CONFIG = {
    "mlflow_tracking_uri": "https://dagshub.com/vishalchauhan91196/MLOps-Production-System.mlflow",
    "dagshub_repo_owner": "vishalchauhan91196",
    "dagshub_repo_name": "MLOps-Production-System"
}

# ========================== SETUP MLflow & DAGSHUB ==========================
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True)
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])


def preprocess_text(text: str) -> str:
    """  Applying text preprocessing to a specific column. """
    try:
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))

        # Remove URLs
        text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
        # Remove numbers
        text = ''.join([char for char in text if not char.isdigit()])
        # Convert to lowercase
        text = text.lower()
        # Remove punctuations
        text = re.sub(f"[{re.escape(string.punctuation)}]", ' ', text)
        # Remove stop words
        text = " ".join([word for word in text.split() if word not in stop_words])
        # Lemmatization
        text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])

        return text
        
    except Exception as e:
        logging.error('Unexpected error during text preprocessing: %s', e)
        raise

# -------------------------------------------------------------------------------------
# Initialize Flask app
app = Flask(__name__)

# Create a custom registry
registry = CollectorRegistry()

# Define your custom metrics using this registry
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)

# -------------------------------------------------------------------------------------
# Model and vectorizer setup
model_name = "LoR_Tfidf_model"

def get_latest_model_version(model_name):
    client         = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

model_version = get_latest_model_version(model_name)
model_uri     = f'models:/{model_name}/{model_version}'

print(f"Fetching model from: {model_uri}")
model      = mlflow.pyfunc.load_model(model_uri)
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))


# Routes
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()
    text = request.form["text"]
    # Clean text
    text = preprocess_text(text)
    # Convert to features
    features    = vectorizer.transform([text])
    features_df = pd.DataFrame(features.toarray(), columns = [str(i) for i in range(features.shape[1])])

    # Predict
    result     = model.predict(features_df)
    prediction = result[0]

    # Increment prediction count metric
    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

    # Measure latency
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    return render_template("index.html", result=prediction)


@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose only custom Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    # app.run(debug=True) # for local use
    app.run(debug=True, host="0.0.0.0", port=5000)  # Accessible from outside Docker    
