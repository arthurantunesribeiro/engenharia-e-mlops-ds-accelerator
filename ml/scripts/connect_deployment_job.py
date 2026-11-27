"""Liga o job `credit_deployment` ao modelo como deployment job (o bundle ainda não faz isso)."""

import json
import subprocess

from mlflow import MlflowClient

MODEL_NAME = "main.credit_ml.credit_score"


def main() -> None:
    """Lê o id do job publicado pelo bundle e o liga ao modelo."""
    summary = json.loads(subprocess.check_output(["databricks", "bundle", "summary", "--output", "json"]))
    job_id = summary["resources"]["jobs"]["deployment"]["id"]
    client = MlflowClient(tracking_uri="databricks", registry_uri="databricks-uc")  # o padrão local é sqlite
    if client.get_registered_model(MODEL_NAME).deployment_job_id == job_id:
        print(f"{MODEL_NAME} já está ligado ao job {job_id}")
        return
    client.update_registered_model(MODEL_NAME, deployment_job_id=job_id)
    print(f"{MODEL_NAME} ligado ao deployment job {job_id}")


if __name__ == "__main__":
    main()
