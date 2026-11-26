"""Deployment job · tarefa 3: promove a versão aprovada e atualiza o endpoint.

Move o alias `champion` (o scoring batch passa a usar a versão na próxima execução) e troca só a versão
servida no endpoint, mantendo o resto da configuração que o bundle declarou. O próximo `bundle deploy`
lê o mesmo champion, então não desfaz a troca.
"""
import argparse

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import ServedEntityInput

ENDPOINT = "credit-score"   # resources/serving.py


def main() -> None:
    """Promove `--model_version` a champion e atualiza o endpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_version", required=True)
    args = parser.parse_args()
    name, version = args.model_name, args.model_version

    mlflow.set_registry_uri("databricks-uc")
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(name, "champion", version)
    if "challenger" in client.get_model_version(name, version).aliases:
        client.delete_registered_model_alias(name, "challenger")
    print(f"champion → v{version}")

    w = WorkspaceClient()
    try:
        config = w.serving_endpoints.get(ENDPOINT).config
    except NotFound:
        config = None
    if config is None:   # primeira versão: o próximo bundle deploy cria o endpoint com o champion
        print(f"{ENDPOINT} ainda não existe: rode o ml-cd para criá-lo com a v{version}")
        return
    entities = [ServedEntityInput.from_dict({**e.as_dict(), "entity_version": str(version)})
                for e in config.served_entities]
    w.serving_endpoints.update_config_and_wait(ENDPOINT, served_entities=entities, traffic_config=config.traffic_config)
    print(f"{ENDPOINT} servindo v{version}")


if __name__ == "__main__":
    main()
