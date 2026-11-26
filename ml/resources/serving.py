"""Endpoint de score, declarado em Python para servir a versão do alias champion."""
from databricks.bundles.core import Bundle, Resources
from databricks.bundles.model_serving_endpoints import ModelServingEndpoint
from databricks.sdk import WorkspaceClient

MODEL_NAME = "main.credit_ml.credit_score"


def load_resources(bundle: Bundle) -> Resources:
    """Declara o endpoint `credit-score` com a versão que tem o alias champion.

    Args:
        bundle: Bundle em deploy (não usado).

    Returns:
        Recursos com o endpoint.
    """
    version = WorkspaceClient().model_versions.get_by_alias(MODEL_NAME, "champion").version

    resources = Resources()
    resources.add_model_serving_endpoint("credit_score_endpoint", ModelServingEndpoint.from_dict({
        "name": "credit-score",
        "config": {
            "served_entities": [{"name": "champion", "entity_name": MODEL_NAME, "entity_version": str(version),
                                 "workload_size": "Small", "scale_to_zero_enabled": True}],
            "traffic_config": {"routes": [{"served_model_name": "champion", "traffic_percentage": 100}]},
        },
        "ai_gateway": {"inference_table_config": {
            "catalog_name": "main", "schema_name": "credit_ml", "table_name_prefix": "credit_score", "enabled": True}},
    }))
    return resources
