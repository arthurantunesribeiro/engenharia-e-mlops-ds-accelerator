# Databricks notebook source
# MAGIC %md
# MAGIC # Consumindo as tabelas geradas
# MAGIC
# MAGIC Scores do batch, quarentena, chamadas ao endpoint e o drift do data profiling, em `credit_ml`.

# COMMAND ----------

from pyspark.sql import functions as F

dbutils.widgets.text("catalog", "workspace", "Catálogo")
spark.sql(f"USE CATALOG {dbutils.widgets.get('catalog')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scores por faixa e segmento

# COMMAND ----------

display(
    spark.table("credit_ml.scores_credito")
    .groupBy("faixa", "segmento")
    .agg(F.count("*").alias("propostas"), F.round(F.avg("score"), 3).alias("score_medio"))
    .orderBy("faixa", "segmento")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quarentena: quais regras do contrato barraram o quê

# COMMAND ----------

display(
    spark.table("credit_ml.propostas_quarentena")
    .select("barrado_em", F.explode("falhas").alias("regra"))
    .groupBy("barrado_em", "regra")
    .count()
    .orderBy(F.desc("barrado_em"), F.desc("count"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chamadas ao endpoint (registradas pelo serviço)

# COMMAND ----------

display(
    spark.table("credit_ml.chamadas_endpoint")
    .groupBy(F.to_date("request_time").alias("dia"), "status_code")
    .agg(F.count("*").alias("chamadas"), F.expr("percentile(latencia_ms, 0.95)").alias("p95_ms"))
    .orderBy("dia", "status_code")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Drift calculado pelo data profiling
# MAGIC PSI de cada coluna contra a baseline, por dia, versão do modelo e segmento. O painel pronto fica
# MAGIC na pasta `credit_score_monitoring` do seu usuário, e na aba **Quality** de `credit_score_inference_log`.

# COMMAND ----------

display(
    spark.table("credit_ml.credit_score_inference_log_drift_metrics")
    .filter("drift_type = 'BASELINE' AND population_stability_index IS NOT NULL")
    .select("window.start", "versao_modelo", "slice_value", "column_name", "population_stability_index")
    .orderBy(F.desc("start"), F.desc("population_stability_index"))
)
