import json
import mlflow
from src.logger import logging
import os
import dagshub

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

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


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = model_info['model_uri']
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        
        logging.info(f'Model {model_name} version {model_version.version} registered.')
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/experiment_model_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "LoR_Tfidf_model"
        register_model(model_name, model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()    