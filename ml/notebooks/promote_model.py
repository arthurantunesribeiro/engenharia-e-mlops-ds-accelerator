# Databricks notebook source
# MAGIC %md
# MAGIC # Operação · Comparar champion e challenger e promover
# MAGIC
# MAGIC O treino registra cada versão nova como `challenger`. Aqui comparamos as duas nas métricas
# MAGIC registradas e, se for o caso, movemos o alias `champion`. O scoring batch passa a usar a nova
# MAGIC versão na próxima execução; o endpoint é atualizado aqui mesmo, e o próximo `bundle deploy`
# MAGIC (que lê o champion) mantém a mesma versão.

# COMMAND ----------

# MAGIC %pip install -q "mlflow==3.17.2" "databricks-sdk==0.155.0"
# MAGIC %restart_python

# COMMAND ----------

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import ServedEntityInput

MODEL_NAME = "main.credit_ml.credit_score"
ENDPOINT = "credit-score"   # resources/serving.py
mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()

# COMMAND ----------

rows = []
for alias in ("champion", "challenger"):
    version = client.get_model_version_by_alias(MODEL_NAME, alias)
    metrics = client.get_run(version.run_id).data.metrics
    rows.append({"alias": alias, "versao": version.version, **{k: round(metrics[k], 4) for k in ("auc", "ks", "log_loss")}})
display(pd.DataFrame(rows))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Promover
# MAGIC Rode só depois de comparar também o tráfego real no monitoramento.

# COMMAND ----------

challenger = client.get_model_version_by_alias(MODEL_NAME, "challenger").version
client.set_registered_model_alias(MODEL_NAME, "champion", challenger)
client.delete_registered_model_alias(MODEL_NAME, "challenger")
print(f"champion → v{challenger}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Atualizar o endpoint
# MAGIC Troca só a versão servida; o resto da configuração continua o que o bundle declarou.

# COMMAND ----------

w = WorkspaceClient()
try:
    config = w.serving_endpoints.get(ENDPOINT).config
except NotFound:
    config = None

if config is None:
    print(f"{ENDPOINT} ainda não existe: o próximo bundle deploy o cria com o champion v{challenger}")
else:
    entities = [ServedEntityInput.from_dict({**e.as_dict(), "entity_version": str(challenger)})
                for e in config.served_entities]
    w.serving_endpoints.update_config_and_wait(ENDPOINT, served_entities=entities,
                                               traffic_config=config.traffic_config)
    print(f"{ENDPOINT} servindo v{challenger}")
