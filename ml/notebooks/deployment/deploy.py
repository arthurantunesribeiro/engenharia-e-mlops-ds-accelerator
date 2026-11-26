# Databricks notebook source
# MAGIC %md
# MAGIC # Deployment job · 3. Deploy
# MAGIC
# MAGIC Move o alias `champion` (o scoring batch passa a usar a versão na próxima execução) e troca só a versão
# MAGIC servida no endpoint, mantendo o resto da configuração que o bundle declarou. O próximo `bundle deploy`
# MAGIC lê o mesmo champion, então não desfaz a troca.

# COMMAND ----------

# MAGIC %pip install -q "mlflow==3.17.2" "databricks-sdk==0.155.0"
# MAGIC %restart_python

# COMMAND ----------

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import ServedEntityInput

ENDPOINT = "credit-score"   # resources/serving.py
model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")

mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()
client.set_registered_model_alias(model_name, "champion", model_version)
if "challenger" in client.get_model_version(model_name, model_version).aliases:
    client.delete_registered_model_alias(model_name, "challenger")
print(f"champion → v{model_version}")

# COMMAND ----------

w = WorkspaceClient()
try:
    config = w.serving_endpoints.get(ENDPOINT).config
except NotFound:
    config = None

if config is None:   # primeira versão: o próximo bundle deploy cria o endpoint com o champion
    print(f"{ENDPOINT} ainda não existe: rode o ml-cd para criá-lo com a v{model_version}")
else:
    entities = [ServedEntityInput.from_dict({**e.as_dict(), "entity_version": str(model_version)})
                for e in config.served_entities]
    w.serving_endpoints.update_config_and_wait(ENDPOINT, served_entities=entities,
                                               traffic_config=config.traffic_config)
    print(f"{ENDPOINT} servindo v{model_version}")
