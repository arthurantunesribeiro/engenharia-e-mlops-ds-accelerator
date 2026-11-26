# Databricks notebook source
# MAGIC %md
# MAGIC # Deployment job · 2. Aprovação
# MAGIC
# MAGIC Falha de propósito na primeira execução. O aprovador compara as versões (run `avaliacao_v<versão>` e
# MAGIC o processo de definição do champion no README) e clica em **Approve** na página da versão, que grava
# MAGIC a tag `approval=Approved` e retoma o job a partir desta tarefa. A chave da tag é o nome da tarefa.

# COMMAND ----------

# MAGIC %pip install -q "mlflow==3.17.2"
# MAGIC %restart_python

# COMMAND ----------

import mlflow

APPROVAL_TAG = "approval"   # = task_key em resources/deployment.yml
model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")

mlflow.set_registry_uri("databricks-uc")
tags = mlflow.MlflowClient().get_model_version(model_name, model_version).tags
if tags.get(APPROVAL_TAG) != "Approved":
    raise Exception(f"{model_name} v{model_version} aguardando aprovação: compare na UI e clique em Approve "
                    "na página da versão")
print(f"v{model_version} aprovada")
