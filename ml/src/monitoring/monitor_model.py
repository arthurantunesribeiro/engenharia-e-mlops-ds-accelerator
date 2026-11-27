"""Monitoramento: refresh do data profiling, qualidade da entrada e alertas de drift.

Em crédito o rótulo só chega meses depois, então o primeiro sinal é o drift. Entrada fora do contrato vem
antes: se a entrada quebrou, o PSI mede o defeito, não o mundo. Os alertas vão para quem diagnostica, sem
retreino automático.
"""
import time

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dataquality import Refresh, RefreshState
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from credit_score.monitoring import build_alerts

ML_SCHEMA = "main.credit_ml"
INFERENCE_LOG = f"{ML_SCHEMA}.credit_score_inference_log"


def refresh_profile() -> None:
    """Pede o recálculo do perfil de inferência e espera terminar.

    Raises:
        AssertionError: Se o refresh não terminar com sucesso.
    """
    w = WorkspaceClient()
    table_id = w.tables.get(INFERENCE_LOG).table_id
    refresh = w.data_quality.create_refresh(object_type="table", object_id=table_id,
                                            refresh=Refresh(object_type="table", object_id=table_id))
    while refresh.state in (RefreshState.MONITOR_REFRESH_STATE_PENDING, RefreshState.MONITOR_REFRESH_STATE_RUNNING):
        time.sleep(30)
        refresh = w.data_quality.get_refresh(object_type="table", object_id=table_id, refresh_id=refresh.refresh_id)
    assert refresh.state == RefreshState.MONITOR_REFRESH_STATE_SUCCESS, f"refresh do perfil: {refresh.state} {refresh.message}"


def out_of_contract_rate(spark: SparkSession) -> float:
    """Calcula a maior taxa fora do contrato entre o endpoint (HTTP 400) e o último lote do batch (quarentena).

    Args:
        spark: Sessão Spark.

    Returns:
        Fração entre 0 e 1.
    """
    calls = spark.table(f"{ML_SCHEMA}.chamadas_endpoint")
    endpoint_rate = calls.filter("status_code = 400").count() / max(calls.count(), 1)
    quarantine = spark.table(f"{ML_SCHEMA}.propostas_quarentena")
    last_batch = quarantine.filter(F.col("barrado_em") == quarantine.agg(F.max("barrado_em")).first()[0]).count()
    batch_rate = last_batch / (last_batch + spark.table(f"{ML_SCHEMA}.scores_credito").count())
    print(f"endpoint: {endpoint_rate:.2%} das chamadas recusadas · batch: {batch_rate:.2%} das propostas em quarentena")
    return max(endpoint_rate, batch_rate)


def latest_drift(spark: SparkSession, version: str) -> pd.DataFrame:
    """Lê o PSI de cada coluna contra a baseline, na janela mais recente da versão informada.

    Args:
        spark: Sessão Spark.
        version: Versão do modelo (a do alias champion).

    Returns:
        Uma linha por coluna e segmento, com `column_name`, `slice_value` e `psi`.
    """
    drift = spark.table(f"{INFERENCE_LOG}_drift_metrics").filter(
        (F.col("drift_type") == "BASELINE") & (F.col("versao_modelo") == version)
        & F.col("population_stability_index").isNotNull())
    latest = drift.agg(F.max("window.start")).first()[0]
    return (drift.filter(F.col("window.start") == latest)
            .select("column_name", "slice_value", F.col("population_stability_index").alias("psi")).toPandas())


def main() -> None:
    """Recalcula o perfil e emite os alertas."""
    spark = SparkSession.builder.getOrCreate()
    mlflow.set_registry_uri("databricks-uc")
    champion = mlflow.MlflowClient().get_model_version_by_alias(f"{ML_SCHEMA}.credit_score", "champion").version

    refresh_profile()
    alerts = build_alerts(latest_drift(spark, champion), out_of_contract_rate(spark))
    for message in alerts:
        print("ALERTA ·", message)
    if not alerts:
        print("Sem alertas: distribuições estáveis")


if __name__ == "__main__":
    main()
