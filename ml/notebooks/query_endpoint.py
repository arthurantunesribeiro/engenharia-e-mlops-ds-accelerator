# Databricks notebook source
# MAGIC %md
# MAGIC # Consumindo o modelo pelo endpoint
# MAGIC
# MAGIC A tela da analista pede o score de uma proposta: o serviço de crédito junta o histórico do
# MAGIC cliente (`features_cliente`) com a proposta e chama o endpoint `credit-score`.
# MAGIC
# MAGIC A primeira célula instala o projeto `ml/` (a pasta acima desta) como pacote, igual aos jobs.

# COMMAND ----------

# MAGIC %pip install -q .. "mlflow==3.17.2" "databricks-sdk==0.155.0"
# MAGIC %restart_python

# COMMAND ----------

import json
import time

import numpy as np
from mlflow.deployments import get_deploy_client

from credit_score.features import INPUT_COLUMNS
from credit_score.service import build_request

ENDPOINT = "credit-score"   # resources/serving.py
client = get_deploy_client("databricks")
customers = spark.table("main.credit_ml.features_cliente")


def customer_history(customer_id: int) -> dict:
    row = customers.filter(f"cliente_id = {int(customer_id)}").first()
    return row.asDict() if row else {}


def request_score(row: dict):
    return client.predict(endpoint=ENDPOINT, inputs={"dataframe_records": [row]})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Uma proposta: três fontes numa requisição

# COMMAND ----------

customer_id = customers.select("cliente_id").first()[0]
row = build_request(customer_history(customer_id), {"valor_solicitado": 15000.0, "prazo_meses": 24}, requests_last_24h=1)
print(request_score(row))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Contrato de entrada: o que o endpoint recusa
# MAGIC A assinatura recusa coluna faltando; o contrato, dentro do modelo, recusa valores fora do domínio.
# MAGIC Nos dois casos, HTTP 400 com o motivo.

# COMMAND ----------

cases = {
    "sem renda_mensal (assinatura)": {k: v for k, v in row.items() if k != "renda_mensal"},
    "uso_limite = 7 (contrato)": {**row, "uso_limite": 7.0},
    "segmento premium (contrato)": {**row, "segmento": "premium"},
}
for case, payload in cases.items():
    try:
        request_score(payload)
        print(f"{case}: aceito")
    except Exception as error:
        print(f"{case}: recusado · {str(error)[-140:]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Latência e tráfego de produção
# MAGIC A primeira chamada depois de um tempo parado inclui o cold start (scale to zero). As chamadas
# MAGIC ficam na tabela de inferência, que o monitoramento usa.

# COMMAND ----------

requests = spark.table("main.credit.gold_propostas_pendentes").select(*INPUT_COLUMNS).limit(200).toPandas()
latencies, rejected = [], 0
for _, r in requests.iterrows():
    start = time.perf_counter()
    try:
        request_score(json.loads(r.to_json()))
        latencies.append((time.perf_counter() - start) * 1000)
    except Exception:
        rejected += 1
print(f"p50 = {np.percentile(latencies, 50):.0f} ms · p95 = {np.percentile(latencies, 95):.0f} ms · "
      f"{rejected} recusada(s) pelo contrato")
