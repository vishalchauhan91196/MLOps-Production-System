import os
import re
import string
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# ========================== CONFIGURATION ==========================
CONFIG = {
    "data_path": "notebooks/data.csv",
    "test_size": 0.2,
    "mlflow_tracking_uri": "https://dagshub.com/vishalchauhan91196/MLOps-Production-System.mlflow",
    "dagshub_repo_owner": "vishalchauhan91196",
    "dagshub_repo_name": "MLOps-Production-System",
    "experiment_name": "LoR Hyperparameter Tuning"
}

# ========================== SETUP MLflow & DAGSHUB ==========================
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True)

mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
mlflow.set_experiment(CONFIG["experiment_name"])

# ========================== TEXT PREPROCESSING ==========================
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return text.lower()

def removing_punctuations(text):
    return re.sub(f"[{re.escape(string.punctuation)}]", ' ', text)

def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def normalize_text(df):
    try:
        df['review'] = df['review'].apply(lower_case)
        df['review'] = df['review'].apply(remove_stop_words)
        df['review'] = df['review'].apply(removing_numbers)
        df['review'] = df['review'].apply(removing_punctuations)
        df['review'] = df['review'].apply(removing_urls)
        df['review'] = df['review'].apply(lemmatization)
        return df

    except Exception as e:
        print(f"Error during text normalization: {e}")
        raise

# ========================== LOAD & PREPARE DATA ==========================
def load_and_prepare_data(file_path):
     """Loads, preprocesses, and vectorizes the dataset."""
     try:
        df = pd.read_csv(file_path)
        df = normalize_text(df)
        df = df[df['sentiment'].isin(['positive', 'negative'])]
        df["sentiment"] = df["sentiment"].map({"negative": 0, "positive": 1})
        
        # Convert text data to TF-IDF vectors
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df["review"])
        y = df["sentiment"]
        
        return train_test_split(X, y, test_size=CONFIG["test_size"], random_state=42), vectorizer
        
     except Exception as e:
        print(f"Error loading data: {e}")
        raise

# ========================== TRAIN & LOG MODEL ==========================
def train_and_log_model(X_train, X_test, y_train, y_test, vectorizer):
    """Trains a Logistic Regression model with GridSearch and logs results to MLflow."""
    
    param_grid = {
        "C": [0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    }

    with mlflow.start_run(run_name="LoR hyperparameter runs"):
        grid_search = GridSearchCV(LogisticRegression(), param_grid, n_jobs=-1, cv=5, scoring='f1')
        grid_search.fit(X_train, y_train)

        # Log all hyperparameter tuning runs
        for params, mean_score, std_score in zip(grid_search.cv_results_["params"],
                                                 grid_search.cv_results_["mean_test_score"],
                                                 grid_search.cv_results_["std_test_score"]):
            with mlflow.start_run(run_name=f"LoR with params: {params}", nested=True):
                model = LogisticRegression(**params)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred),
                    "mean_cv_score": mean_score,
                    "std_cv_score": std_score
                }

                # Log parameters & metrics
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)

                print(f"Params: {params} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")


        # Log the best model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_f1 = grid_search.best_score_

        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_score", best_f1)
        mlflow.sklearn.log_model(best_model, "model")
    
        print(f"\nBest Params: {best_params} | Best F1 Score: {best_f1:.4f}")


# ==========================
# Main Execution
# ==========================
if __name__ == "__main__":
    (X_train, X_test, y_train, y_test), vectorizer = load_and_prepare_data(CONFIG["data_path"])
    train_and_log_model(X_train, X_test, y_train, y_test, vectorizer)