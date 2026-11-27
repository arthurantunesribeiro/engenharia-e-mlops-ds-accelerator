"""Monitoramento: recalcula o perfil de inferência e avisa no Google Chat.

O data profiling calcula o drift contra a baseline do treino (tabelas `_profile_metrics` e `_drift_metrics`
e o painel). Depois do refresh, este job roda as mesmas consultas dos SQL alerts (`resources/alerts/`) e,
se algum limite foi ultrapassado, posta no Google Chat. O webhook fica no secret
`credit_ml/google_chat_webhook`; o Chat não aceita o webhook genérico dos notification destinations do
Databricks, que manda um JSON sem o campo `text`. Quem decide é o processo de definição do champion, sem
retreino automático.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dataquality import Refresh, RefreshState
from pyspark.sql import SparkSession

from credit_score.monitoring import chat_message

INFERENCE_LOG = "main.credit_ml.credit_score_inference_log"
SECRET_SCOPE, WEBHOOK_KEY = "credit_ml", "google_chat_webhook"


def refresh_profile(w: WorkspaceClient) -> None:
    """Pede o recálculo do perfil de inferência e espera terminar.

    Args:
        w: Cliente do workspace.

    Raises:
        AssertionError: Se o refresh não terminar com sucesso.
    """
    table_id = w.tables.get(INFERENCE_LOG).table_id
    refresh = w.data_quality.create_refresh(object_type="table", object_id=table_id,
                                            refresh=Refresh(object_type="table", object_id=table_id))
    while refresh.state in (RefreshState.MONITOR_REFRESH_STATE_PENDING, RefreshState.MONITOR_REFRESH_STATE_RUNNING):
        time.sleep(30)
        refresh = w.data_quality.get_refresh(object_type="table", object_id=table_id, refresh_id=refresh.refresh_id)
    assert refresh.state == RefreshState.MONITOR_REFRESH_STATE_SUCCESS, f"refresh do perfil: {refresh.state} {refresh.message}"
    print(f"perfil de {INFERENCE_LOG} recalculado")


def query_row(spark: SparkSession, path: Path) -> dict:
    """Roda uma consulta de alerta e devolve a única linha.

    Args:
        spark: Sessão Spark.
        path: Arquivo `.sql`.

    Returns:
        A linha como dicionário (vazio se a consulta não devolver nada).
    """
    row = spark.sql(path.read_text()).first()
    return row.asDict() if row else {}


def post_to_chat(w: WorkspaceClient, text: str) -> None:
    """Posta uma mensagem no Google Chat pelo webhook guardado no secret.

    Args:
        w: Cliente do workspace.
        text: Mensagem.
    """
    url = w.dbutils.secrets.get(SECRET_SCOPE, WEBHOOK_KEY)
    request = urllib.request.Request(url, data=json.dumps({"text": text}).encode(),
                                     headers={"Content-Type": "application/json; charset=UTF-8"})
    with urllib.request.urlopen(request, timeout=30) as response:
        print(f"Google Chat: HTTP {response.status}")


def main() -> None:
    """Recalcula o perfil, roda as consultas dos alertas e avisa no Chat se algum limite foi ultrapassado."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, help="pasta resources/alerts publicada pelo bundle")
    queries = Path(parser.parse_args().queries)

    w = WorkspaceClient()
    refresh_profile(w)
    spark = SparkSession.builder.getOrCreate()
    drift, out_of_contract = query_row(spark, queries / "drift.sql"), query_row(spark, queries / "fora_do_contrato.sql")
    print(f"drift: {drift} · fora do contrato: {out_of_contract}")
    message = chat_message(drift, out_of_contract)
    if message is None:
        print("Sem alertas: dentro dos limites")
        return
    print(message)
    post_to_chat(w, message)


if __name__ == "__main__":
    main()
