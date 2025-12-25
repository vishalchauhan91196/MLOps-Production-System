import os
import mlflow
import dagshub

# ========================== CONFIGURATION ==========================
CONFIG = {
    "mlflow_tracking_uri": "https://dagshub.com/vishalchauhan91196/MLOps-Production-System.mlflow",
    "dagshub_repo_owner": "vishalchauhan91196",
    "dagshub_repo_name": "MLOps-Production-System"
}

# ========================== SETUP MLflow & DAGSHUB ==========================
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True)
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])


model_name = "LoR_Tfidf_model"

def promote_model():
    client = mlflow.MlflowClient()

    # Get the latest version in staging
    latest_version_staging = client.get_latest_versions(model_name, stages=["Staging"])[0].version

    # Archive the current production model
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    # Promote the new model to production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production"
    )
    print(f"Model version {latest_version_staging} promoted to Production")

if __name__ == "__main__":
    promote_model()    