"""Endpoint de score com a versão do alias champion (sem inference table no Free Edition)."""

from databricks.bundles.core import Bundle, Resources
from databricks.bundles.model_serving_endpoints import ModelServingEndpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound

MODEL_NAME = "main.credit_ml.credit_score"


def load_resources(bundle: Bundle) -> Resources:
    """Declara o endpoint `credit-score` com a versão que tem o alias champion.

    Args:
        bundle: Bundle em deploy (não usado).

    Returns:
        Recursos com o endpoint, ou nenhum recurso se ainda não houver champion.
    """
    resources = Resources()
    try:
        version = WorkspaceClient().model_versions.get_by_alias(MODEL_NAME, "champion").version
    except NotFound:
        resources.add_diagnostic_warning(
            f"{MODEL_NAME} ainda não tem o alias champion: o endpoint fica para o deploy depois do primeiro treino"
        )
        return resources

    resources.add_model_serving_endpoint(
        "credit_score_endpoint",
        ModelServingEndpoint.from_dict(
            {
                "name": "credit-score",
                "config": {
                    "served_entities": [
                        {
                            "name": "champion",
                            "entity_name": MODEL_NAME,
                            "entity_version": str(version),
                            "workload_size": "Small",
                            "scale_to_zero_enabled": True,
                        }
                    ],
                    "traffic_config": {"routes": [{"served_model_name": "champion", "traffic_percentage": 100}]},
                },
            }
        ),
    )
    return resources
