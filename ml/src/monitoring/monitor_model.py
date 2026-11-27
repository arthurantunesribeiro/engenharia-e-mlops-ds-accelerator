"""Monitoramento: recalcula o perfil de inferência depois que o modelo pontua."""

import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dataquality import Refresh, RefreshState

from credit_score.settings import catalog_from_args


def refresh_profile(catalog: str) -> None:
    """Pede o recálculo do perfil de inferência e espera terminar.

    Args:
        catalog: Catálogo do schema `credit_ml`.

    Raises:
        AssertionError: Se o refresh não terminar com sucesso.
    """
    inference_log = f"{catalog}.credit_ml.credit_score_inference_log"
    w = WorkspaceClient()
    table_id = w.tables.get(inference_log).table_id
    refresh = w.data_quality.create_refresh(
        object_type="table", object_id=table_id, refresh=Refresh(object_type="table", object_id=table_id)
    )
    while refresh.state in (RefreshState.MONITOR_REFRESH_STATE_PENDING, RefreshState.MONITOR_REFRESH_STATE_RUNNING):
        time.sleep(30)
        refresh = w.data_quality.get_refresh(object_type="table", object_id=table_id, refresh_id=refresh.refresh_id)
    assert refresh.state == RefreshState.MONITOR_REFRESH_STATE_SUCCESS, (
        f"refresh do perfil: {refresh.state} {refresh.message}"
    )
    print(f"perfil de {inference_log} recalculado")


if __name__ == "__main__":
    refresh_profile(catalog_from_args())
