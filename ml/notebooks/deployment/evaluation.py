# Databricks notebook source
# MAGIC %md
# MAGIC # Deployment job · 1. Avaliação
# MAGIC
# MAGIC Cada treino valida numa janela diferente, então as runs de treino não se comparam direito. Aqui a
# MAGIC versão nova e o champion pontuam a mesma validação (a janela mais recente da ABT atual), pelo mesmo
# MAGIC caminho do endpoint (o PyFunc registrado). Tudo fica na run de origem da versão (a run de treino que a
# MAGIC gerou), com o prefixo `deploy_`, e as métricas da versão nova também no modelo logado, que aparece na
# MAGIC página da versão no Unity Catalog. Os parâmetros `model_name` e `model_version` vêm do job.

# COMMAND ----------

# MAGIC %pip install -q ../.. "mlflow==3.17.2"
# MAGIC %restart_python

# COMMAND ----------

import mlflow

from credit_score.evaluation import score_report, split_in_time
from credit_score.features import INPUT_COLUMNS

model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")
spark.sql(f"USE CATALOG {model_name.split('.')[0]}")

mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()

# COMMAND ----------

versions = {"nova": client.get_model_version(model_name, model_version)}
try:
    champion = client.get_model_version_by_alias(model_name, "champion")
    if champion.version != versions["nova"].version:
        versions["champion"] = champion
except mlflow.exceptions.MlflowException:
    pass

_, valid = split_in_time(spark.table("credit.gold_abt_credito").toPandas())

# COMMAND ----------

with mlflow.start_run(run_id=versions["nova"].run_id):
    mlflow.set_tags(
        {
            "deploy_champion_version": versions["champion"].version if "champion" in versions else "nenhum",
            "deploy_janela": f"{valid['data_ref'].min()} a {valid['data_ref'].max()} ({len(valid)} linhas)",
        }
    )
    for role, mv in versions.items():
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/{mv.version}")
        p = model.predict(valid[INPUT_COLUMNS])["score"].to_numpy()
        metrics, by_segment, bands = score_report(valid, p)
        mlflow.log_metrics({f"deploy_{role}_{k}": v for k, v in metrics.items()}, model_id=versions["nova"].model_id)
        mlflow.log_table(by_segment, f"deploy/{role}/metrics_by_segment.json")
        mlflow.log_table(bands, f"deploy/{role}/faixas.json")
        print(f"{role} v{mv.version}: " + " · ".join(f"{k} {v:.4f}" for k, v in metrics.items()))
