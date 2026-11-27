"""Job de monitoramento: só é declarado quando as tabelas do gatilho existem."""

from databricks.bundles.core import Bundle, Resources
from databricks.bundles.jobs import Job
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound

TRIGGER_TABLES = ["main.credit_ml.scores_credito", "main.credit_ml.chamadas_endpoint"]


def load_resources(bundle: Bundle) -> Resources:
    """Declara o job que mede drift e qualidade da entrada quando o modelo pontua.

    Args:
        bundle: Bundle em deploy (não usado).

    Returns:
        Recursos com o job, ou nenhum recurso se alguma tabela do gatilho ainda não existir.
    """
    resources = Resources()
    w = WorkspaceClient()
    missing = []
    for table in TRIGGER_TABLES:
        try:
            w.tables.get(table)
        except NotFound:
            missing.append(table)
    if missing:
        resources.add_diagnostic_warning(
            f"{', '.join(missing)} ainda não existe(m): o job de monitoramento fica para o deploy depois do "
            "primeiro scoring batch e das primeiras chamadas ao endpoint"
        )
        return resources

    def task(key: str, script: str, parameters: tuple = (), **kw) -> dict:
        return {
            "task_key": key,
            "environment_key": "default",
            **kw,
            "spark_python_task": {
                "python_file": f"${{workspace.file_path}}/src/monitoring/{script}",
                "parameters": list(parameters),
            },
        }

    resources.add_resource(
        "monitoring",
        Job.from_dict(
            {
                "name": "credit_monitoring",
                "trigger": {
                    "table_update": {
                        "table_names": TRIGGER_TABLES,
                        "condition": "ANY_UPDATED",
                        "min_time_between_triggers_seconds": 3600,
                    }
                },
                "environments": [{"environment_key": "default", "spec": "${var.job_environment}"}],
                "tasks": [
                    task("build_inference_log", "build_inference_log.py"),
                    task("monitor_model", "monitor_model.py", depends_on=[{"task_key": "build_inference_log"}]),
                ],
            }
        ),
    )
    return resources
