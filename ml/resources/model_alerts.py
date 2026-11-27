"""SQL alerts do monitoramento, como a documentação do data profiling recomenda.

Cada alerta roda uma consulta de `resources/alerts/` no SQL warehouse e avisa o destino de notificação
(secret `credit_ml/alert_destination_id`, um e-mail) quando o valor passa do limite. As mesmas consultas
e limites vão para o Google Chat pelo job de monitoramento (`src/monitoring/monitor_model.py`):

- drift: maior PSI contra a baseline do treino, na janela mais recente do perfil de inferência;
- entrada fora do contrato: chamadas recusadas pelo endpoint ou propostas em quarentena no último lote.

Em crédito o rótulo só chega meses depois, então o primeiro sinal é o drift. Entrada fora do contrato vem
antes: se a entrada quebrou, o PSI mede o defeito, não o mundo. Os alertas vão para quem diagnostica, sem
retreino automático.
"""
from pathlib import Path

from databricks.bundles.alerts import Alert
from databricks.bundles.core import Bundle, Resources

from resources.notifications import secret

QUERIES = Path(__file__).parent / "alerts"
SCHEDULE = {"quartz_cron_schedule": "0 0 8 * * ?", "timezone_id": "America/Sao_Paulo"}   # todo dia, 8h
ALERTS = {   # limites iguais aos de credit_score.monitoring (DRIFT_LIMIT, OUT_OF_CONTRACT_LIMIT)
    # chave: (arquivo, coluna avaliada, limite, nome)
    "credit_score_drift": ("drift.sql", "maior_psi", 0.2, "credit_score · drift acima de 0,2 (PSI)"),
    "credit_score_fora_do_contrato": ("fora_do_contrato.sql", "maior_taxa_fora_do_contrato", 0.02,
                                      "credit_score · entrada fora do contrato acima de 2%"),
}
# Template da notificação (variáveis {{...}} do SQL alert; HTML só aparece formatado no e-mail). A tabela traz
# todas as colunas da consulta: no drift, onde está o maior PSI e quantas colunas passaram do limite.
SUMMARY = "[credit_score] {{ALERT_NAME}}: {{ALERT_STATUS}}"
DESCRIPTION = """<p><b>{{ALERT_COLUMN}} = {{QUERY_RESULT_VALUE}}</b> ({{ALERT_CONDITION}} {{ALERT_THRESHOLD}})</p>
{{QUERY_RESULT_TABLE}}
<p><a href="{{ALERT_URL}}">Abrir o alerta</a></p>"""


def load_resources(bundle: Bundle) -> Resources:
    """Declara os SQL alerts de drift e de entrada fora do contrato.

    Args:
        bundle: Bundle em deploy (não usado).

    Returns:
        Recursos com os alertas.
    """
    resources = Resources()
    for key, (query_file, column, threshold, name) in ALERTS.items():
        resources.add_resource(key, Alert.from_dict({
            "display_name": name,
            "warehouse_id": "${var.warehouse_id}",
            "query_text": (QUERIES / query_file).read_text(),
            "schedule": SCHEDULE,
            "custom_summary": SUMMARY,
            "custom_description": DESCRIPTION,
            "evaluation": {
                "source": {"name": column},
                "comparison_operator": "GREATER_THAN",
                "threshold": {"value": {"double_value": threshold}},
                "empty_result_state": "OK",   # sem dado ainda não é alerta
                "notification": {"notify_on_ok": False,
                                 "subscriptions": [{"destination_id": secret("alert_destination_id")}]},
            },
        }))
    return resources
