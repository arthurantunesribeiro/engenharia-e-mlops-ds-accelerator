"""Liga o job `credit_deployment` ao modelo como deployment job (o bundle ainda não faz isso)."""

import json
import subprocess

from mlflow import MlflowClient


def main() -> None:
    """Lê o id do job publicado pelo bundle e o liga ao modelo."""
    summary = json.loads(subprocess.check_output(["databricks", "bundle", "summary", "--output", "json"]))
    job = summary["resources"]["jobs"]["deployment"]
    job_id = job["id"]
    model_name = next(p["default"] for p in job["parameters"] if p["name"] == "model_name")
    client = MlflowClient(tracking_uri="databricks", registry_uri="databricks-uc")  # o padrão local é sqlite
    if client.get_registered_model(model_name).deployment_job_id == job_id:
        print(f"{model_name} já está ligado ao job {job_id}")
        return
    client.update_registered_model(model_name, deployment_job_id=job_id)
    print(f"{model_name} ligado ao deployment job {job_id}")


if __name__ == "__main__":
    main()
