"""SQL alerts de drift e de entrada fora do contrato, avisados por e-mail."""

from pathlib import Path

from databricks.bundles.alerts import Alert
from databricks.bundles.core import Bundle, Resources

from resources.notifications import secret

QUERIES = Path(__file__).parent / "alerts"
SCHEDULE = {"quartz_cron_schedule": "0 0 8 * * ?", "timezone_id": "America/Sao_Paulo"}
ALERTS = {
    "credit_score_drift": ("drift.sql", "maior_psi", 0.2, "credit_score · drift acima de 0,2 (PSI)"),
    "credit_score_fora_do_contrato": (
        "fora_do_contrato.sql",
        "maior_taxa_fora_do_contrato",
        0.02,
        "credit_score · entrada fora do contrato acima de 2%",
    ),
}
SUMMARY = "[credit_score] {{ALERT_NAME}}: {{ALERT_STATUS}}"
DESCRIPTION = """<p><b>{{ALERT_COLUMN}} = {{QUERY_RESULT_VALUE}}</b> ({{ALERT_CONDITION}} {{ALERT_THRESHOLD}})</p>
{{QUERY_RESULT_TABLE}}
<p><a href="{{ALERT_URL}}">Abrir o alerta</a></p>"""


def load_resources(bundle: Bundle) -> Resources:
    """Declara os SQL alerts de drift e de entrada fora do contrato.

    Args:
        bundle: Bundle em deploy, com a variável `catalog`.

    Returns:
        Recursos com os alertas.
    """
    resources = Resources()
    for key, (query_file, column, threshold, name) in ALERTS.items():
        resources.add_resource(
            key,
            Alert.from_dict(
                {
                    "display_name": name,
                    "warehouse_id": "${var.warehouse_id}",
                    "query_text": (QUERIES / query_file).read_text().replace("{catalog}", bundle.variables["catalog"]),
                    "schedule": SCHEDULE,
                    "custom_summary": SUMMARY,
                    "custom_description": DESCRIPTION,
                    "evaluation": {
                        "source": {"name": column},
                        "comparison_operator": "GREATER_THAN",
                        "threshold": {"value": {"double_value": threshold}},
                        "empty_result_state": "OK",
                        "notification": {
                            "notify_on_ok": False,
                            "subscriptions": [{"destination_id": secret("alert_destination_id")}],
                        },
                    },
                }
            ),
        )
    return resources
