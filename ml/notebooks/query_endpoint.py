# Databricks notebook source
# MAGIC %md
# MAGIC # Consumindo o modelo pelo endpoint
# MAGIC
# MAGIC A tela da analista pede o score de uma proposta: o serviço de crédito junta o histórico do
# MAGIC cliente (`features_cliente`) com a proposta e chama o endpoint `credit-score`.
# MAGIC
# MAGIC O serviço também registra cada chamada (entradas, score, status, latência e versão servida) em
# MAGIC `credit_ml.chamadas_endpoint`, que o monitoramento usa. Uma inference table do Model Serving
# MAGIC faria esse registro sozinha, para qualquer cliente, mas não está disponível no Databricks Free Edition.
# MAGIC
# MAGIC A primeira célula instala o projeto `ml/` (a pasta acima desta) como pacote, igual aos jobs.

# COMMAND ----------

# MAGIC %pip install -q .. "mlflow==3.17.2" "databricks-sdk==0.155.0"
# MAGIC %restart_python

# COMMAND ----------

import json
import time
from datetime import datetime

import numpy as np
from databricks.sdk import WorkspaceClient
from mlflow.deployments import get_deploy_client

from credit_score.features import INPUT_COLUMNS
from credit_score.service import CALL_LOG_SCHEMA, build_request, call_record, status_code_of

ENDPOINT = "credit-score"
CALL_LOG = "credit_ml.chamadas_endpoint"
client = get_deploy_client("databricks")
dbutils.widgets.text("catalog", "workspace", "Catálogo")
spark.sql(f"USE CATALOG {dbutils.widgets.get('catalog')}")
customers = spark.table("credit_ml.features_cliente")
served_version = WorkspaceClient().serving_endpoints.get(ENDPOINT).config.served_entities[0].entity_version
calls = []


def customer_history(customer_id: int) -> dict:
    row = customers.filter(f"cliente_id = {int(customer_id)}").first()
    return row.asDict() if row else {}


def request_score(row: dict):
    """Chama o endpoint e registra a chamada, aceita ou recusada."""
    started, start = datetime.now(), time.perf_counter()
    try:
        out = client.predict(endpoint=ENDPOINT, inputs={"dataframe_records": [row]})
    except Exception as error:
        calls.append(
            call_record(
                row,
                request_time=started,
                status_code=status_code_of(error),
                latency_ms=(time.perf_counter() - start) * 1000,
                model_version=served_version,
                error=str(error)[-500:],
            )
        )
        raise
    calls.append(
        call_record(
            row,
            request_time=started,
            status_code=200,
            latency_ms=(time.perf_counter() - start) * 1000,
            model_version=served_version,
            prediction=out["predictions"][0],
        )
    )
    return out


# COMMAND ----------

# MAGIC %md
# MAGIC ### Uma proposta: três fontes numa requisição

# COMMAND ----------

customer_id = customers.select("cliente_id").first()[0]
row = build_request(
    customer_history(customer_id), {"valor_solicitado": 15000.0, "prazo_meses": 24}, requests_last_24h=1
)
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
# MAGIC A primeira chamada depois de um tempo parado inclui o cold start (scale to zero).

# COMMAND ----------

requests = spark.table("credit.gold_propostas_pendentes").select(*INPUT_COLUMNS).limit(200).toPandas()
latencies, rejected = [], 0
for _, r in requests.iterrows():
    start = time.perf_counter()
    try:
        request_score(json.loads(r.to_json()))
        latencies.append((time.perf_counter() - start) * 1000)
    except Exception:
        rejected += 1
print(
    f"p50 = {np.percentile(latencies, 50):.0f} ms · p95 = {np.percentile(latencies, 95):.0f} ms · "
    f"{rejected} recusada(s) pelo contrato"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Registrar as chamadas
# MAGIC O serviço grava o que chamou; a nova linha na tabela dispara o job de monitoramento.

# COMMAND ----------

(spark.createDataFrame(calls, schema=CALL_LOG_SCHEMA).write.mode("append").saveAsTable(CALL_LOG))
print(f"{len(calls)} chamadas registradas em {CALL_LOG}")
