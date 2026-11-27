"""Data profiling da tabela de inferência: drift das entradas e do score, por versão e segmento."""
from databricks.bundles.core import Bundle, Resources
from databricks.bundles.data_quality_monitors import DataQualityMonitor
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound

from resources.notifications import secret

INFERENCE_LOG = "main.credit_ml.credit_score_inference_log"


def load_resources(bundle: Bundle) -> Resources:
    """Declara o perfil de inferência sobre a tabela montada pelo job de monitoramento.

    Args:
        bundle: Bundle em deploy (não usado).

    Returns:
        Recursos com o monitor de data profiling, ou nenhum recurso se a tabela ainda não existir.
    """
    resources = Resources()
    try:
        table_id = WorkspaceClient().tables.get(INFERENCE_LOG).table_id
    except NotFound:   # a tabela nasce na primeira execução do job de monitoramento
        resources.add_diagnostic_warning(
            f"{INFERENCE_LOG} ainda não existe: o perfil de inferência fica para o deploy depois do monitoramento")
        return resources

    resources.add_data_quality_monitor("credit_score_inference_profile", DataQualityMonitor.from_dict({
        "object_type": "table",
        "object_id": table_id,
        "data_profiling_config": {
            "output_schema_id": "${resources.schemas.credit_ml.schema_id}",
            "assets_dir": "/Workspace/Users/${workspace.current_user.userName}/credit_score_monitoring",
            "inference_log": {
                "problem_type": "INFERENCE_PROBLEM_TYPE_REGRESSION",   # o score é uma probabilidade contínua
                "prediction_column": "score",
                "timestamp_column": "request_time",
                "model_id_column": "versao_modelo",
                "granularities": ["AGGREGATION_GRANULARITY_1_DAY"],
            },
            "baseline_table_name": "main.credit_ml.credit_score_baseline",
            "slicing_exprs": ["segmento"],
            "notification_settings": {"on_failure": {"email_addresses": [secret("alert_email")]}},   # refresh falhou
        },
    }))
    return resources
