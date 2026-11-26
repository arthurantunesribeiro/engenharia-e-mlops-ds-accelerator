"""Deployment job · tarefa 1: avalia a versão nova e o champion na mesma janela.

Cada treino valida numa janela diferente, então as runs de treino não se comparam direito. Aqui as duas
versões pontuam a mesma validação (a janela mais recente da ABT atual), pelo mesmo caminho do endpoint (o
PyFunc registrado). As métricas da versão nova vão para a página dela no Unity Catalog; a comparação
completa fica na run `avaliacao_v<versão>`, ligada à versão pela tag `avaliacao_run`.
"""
import argparse

import mlflow
from pyspark.sql import SparkSession

from credit_score.evaluation import score_report, split_in_time
from credit_score.features import INPUT_COLUMNS

ABT_TABLE = "main.credit.gold_abt_credito"


def main() -> None:
    """Avalia `--model_version` e o champion atual e loga a comparação."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_version", required=True)
    args = parser.parse_args()

    spark = SparkSession.builder.getOrCreate()
    mlflow.set_registry_uri("databricks-uc")
    user = spark.sql("SELECT current_user()").first()[0]
    mlflow.set_experiment(f"/Users/{user}/credit_score")   # resources/foundation.yml
    client = mlflow.MlflowClient()

    versions = {"nova": client.get_model_version(args.model_name, args.model_version)}
    try:
        champion = client.get_model_version_by_alias(args.model_name, "champion")
        if champion.version != versions["nova"].version:
            versions["champion"] = champion
    except mlflow.exceptions.MlflowException:   # primeira versão: não há com quem comparar
        pass

    _, valid = split_in_time(spark.table(ABT_TABLE).toPandas())
    with mlflow.start_run(run_name=f"avaliacao_v{args.model_version}") as run:
        mlflow.set_tags({"model_name": args.model_name, "model_version": args.model_version,
                         "champion_version": versions["champion"].version if "champion" in versions else "nenhum"})
        mlflow.log_params({"janela_inicio": str(valid["data_ref"].min()), "janela_fim": str(valid["data_ref"].max()),
                           "linhas": len(valid)})
        for role, mv in versions.items():
            model = mlflow.pyfunc.load_model(f"models:/{args.model_name}/{mv.version}")
            p = model.predict(valid[INPUT_COLUMNS])["score"].to_numpy()
            metrics, by_segment, bands = score_report(valid, p)
            mlflow.log_metrics({f"{role}_{k}": v for k, v in metrics.items()})
            mlflow.log_table(by_segment, f"{role}/metrics_by_segment.json")
            mlflow.log_table(bands, f"{role}/faixas.json")
            if role == "nova" and mv.model_id:   # aparece na página da versão
                mlflow.log_metrics(metrics, model_id=mv.model_id)
            print(f"{role} v{mv.version}: " + " · ".join(f"{k} {v:.4f}" for k, v in metrics.items()))
    client.set_model_version_tag(args.model_name, args.model_version, "avaliacao_run", run.info.run_id)


if __name__ == "__main__":
    main()
