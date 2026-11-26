"""Deployment job · tarefa 2: só segue com a versão aprovada na UI.

Falha de propósito na primeira execução. O aprovador compara as versões (run `avaliacao_v<versão>` e o
processo de definição do champion no README) e clica em *Approve* na página da versão, que grava a tag
`approval=Approved` e retoma o job a partir desta tarefa. A chave da tag é o nome da tarefa (`approval`).
"""
import argparse

import mlflow

APPROVAL_TAG = "approval"   # = task_key em resources/deployment.yml


def main() -> None:
    """Confere a tag de aprovação da versão.

    Raises:
        RuntimeError: Se a versão ainda não foi aprovada.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_version", required=True)
    args = parser.parse_args()

    mlflow.set_registry_uri("databricks-uc")
    tags = mlflow.MlflowClient().get_model_version(args.model_name, args.model_version).tags
    if tags.get(APPROVAL_TAG) != "Approved":
        raise RuntimeError(f"{args.model_name} v{args.model_version} aguardando aprovação: compare na UI e clique "
                           "em Approve na página da versão")
    print(f"v{args.model_version} aprovada")


if __name__ == "__main__":
    main()
