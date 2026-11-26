"""Liga o job `credit_deployment` ao modelo no Unity Catalog, como deployment job.

O Databricks Asset Bundles ainda não tem esse campo no registered model. Roda no ml-cd depois do
`databricks bundle deploy` (precisa de MANAGE no modelo e de CAN MANAGE RUN no job).
"""
import json
import subprocess

from mlflow import MlflowClient

MODEL_NAME = "main.credit_ml.credit_score"


def main() -> None:
    """Lê o id do job publicado pelo bundle e o liga ao modelo."""
    summary = json.loads(subprocess.check_output(["databricks", "bundle", "summary", "--output", "json"]))
    job_id = summary["resources"]["jobs"]["deployment"]["id"]
    client = MlflowClient(registry_uri="databricks-uc")
    if client.get_registered_model(MODEL_NAME).deployment_job_id == job_id:
        print(f"{MODEL_NAME} já está ligado ao job {job_id}")
        return
    client.update_registered_model(MODEL_NAME, deployment_job_id=job_id)
    print(f"{MODEL_NAME} ligado ao deployment job {job_id}")


if __name__ == "__main__":
    main()
