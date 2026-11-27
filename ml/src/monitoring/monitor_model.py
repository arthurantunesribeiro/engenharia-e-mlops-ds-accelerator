"""Monitoramento: recalcula o perfil de inferência depois que o modelo pontua."""

import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dataquality import Refresh, RefreshState

INFERENCE_LOG = "main.credit_ml.credit_score_inference_log"


def refresh_profile() -> None:
    """Pede o recálculo do perfil de inferência e espera terminar.

    Raises:
        AssertionError: Se o refresh não terminar com sucesso.
    """
    w = WorkspaceClient()
    table_id = w.tables.get(INFERENCE_LOG).table_id
    refresh = w.data_quality.create_refresh(
        object_type="table", object_id=table_id, refresh=Refresh(object_type="table", object_id=table_id)
    )
    while refresh.state in (RefreshState.MONITOR_REFRESH_STATE_PENDING, RefreshState.MONITOR_REFRESH_STATE_RUNNING):
        time.sleep(30)
        refresh = w.data_quality.get_refresh(object_type="table", object_id=table_id, refresh_id=refresh.refresh_id)
    assert refresh.state == RefreshState.MONITOR_REFRESH_STATE_SUCCESS, (
        f"refresh do perfil: {refresh.state} {refresh.message}"
    )
    print(f"perfil de {INFERENCE_LOG} recalculado")


if __name__ == "__main__":
    refresh_profile()
