# Databricks notebook source
# MAGIC %md
# MAGIC # Deployment job · 1. Avaliação
# MAGIC
# MAGIC Cada treino valida numa janela diferente, então as runs de treino não se comparam direito. Aqui a
# MAGIC versão nova e o champion pontuam a mesma validação (a janela mais recente da ABT atual), pelo mesmo
# MAGIC caminho do endpoint (o PyFunc registrado). As métricas da versão nova vão para a página dela no Unity
# MAGIC Catalog; a comparação completa fica na run `avaliacao_v<versão>`, ligada à versão pela tag
# MAGIC `avaliacao_run`. Os parâmetros `model_name` e `model_version` vêm do job.

# COMMAND ----------

# MAGIC %pip install -q ../.. "mlflow==3.17.2"
# MAGIC %restart_python

# COMMAND ----------

import mlflow

from credit_score.evaluation import score_report, split_in_time
from credit_score.features import INPUT_COLUMNS

ABT_TABLE = "main.credit.gold_abt_credito"
model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")

mlflow.set_registry_uri("databricks-uc")
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/credit_score")   # resources/foundation.yml
client = mlflow.MlflowClient()

# COMMAND ----------

versions = {"nova": client.get_model_version(model_name, model_version)}
try:
    champion = client.get_model_version_by_alias(model_name, "champion")
    if champion.version != versions["nova"].version:
        versions["champion"] = champion
except mlflow.exceptions.MlflowException:   # primeira versão: não há com quem comparar
    pass

_, valid = split_in_time(spark.table(ABT_TABLE).toPandas())

# COMMAND ----------

with mlflow.start_run(run_name=f"avaliacao_v{model_version}") as run:
    mlflow.set_tags({"model_name": model_name, "model_version": model_version,
                     "champion_version": versions["champion"].version if "champion" in versions else "nenhum"})
    mlflow.log_params({"janela_inicio": str(valid["data_ref"].min()), "janela_fim": str(valid["data_ref"].max()),
                       "linhas": len(valid)})
    for role, mv in versions.items():
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/{mv.version}")
        p = model.predict(valid[INPUT_COLUMNS])["score"].to_numpy()
        metrics, by_segment, bands = score_report(valid, p)
        mlflow.log_metrics({f"{role}_{k}": v for k, v in metrics.items()})
        mlflow.log_table(by_segment, f"{role}/metrics_by_segment.json")
        mlflow.log_table(bands, f"{role}/faixas.json")
        if role == "nova" and mv.model_id:   # aparece na página da versão
            mlflow.log_metrics(metrics, model_id=mv.model_id)
        print(f"{role} v{mv.version}: " + " · ".join(f"{k} {v:.4f}" for k, v in metrics.items()))

client.set_model_version_tag(model_name, model_version, "avaliacao_run", run.info.run_id)
