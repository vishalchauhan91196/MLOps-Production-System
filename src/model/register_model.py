import json
import mlflow
from src.logger import logging
import os
import dagshub

# ========================== CONFIGURATION ==========================
CONFIG = {
    "data_path": "notebooks/data.csv",
    "test_size": 0.2,
    "mlflow_tracking_uri": "https://dagshub.com/vishalchauhan91196/MLOps-Production-System.mlflow",
    "dagshub_repo_owner": "vishalchauhan91196",
    "dagshub_repo_name": "MLOps-Production-System"
}

# ========================== SETUP MLflow & DAGSHUB ==========================
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True)
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])

data_path = os.path.join(from_root(), 'data')
model_path = os.path.join(from_root(), 'models')