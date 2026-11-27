"""Destinatários dos avisos, lidos do secret scope `credit_ml` no deploy."""

import base64
import functools
from dataclasses import replace

from databricks.bundles.core import Bundle, job_mutator
from databricks.bundles.jobs import Job, JobEmailNotifications
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound

SECRET_SCOPE = "credit_ml"
NO_FAILURE_EMAIL = {"credit_deployment"}


@functools.cache
def secret(key: str) -> str:
    """Lê um secret do scope `credit_ml`.

    Args:
        key: Chave do secret.

    Returns:
        O valor, em texto.

    Raises:
        RuntimeError: Se o secret não existir, com o comando que o cria.
    """
    try:
        value = WorkspaceClient().secrets.get_secret(SECRET_SCOPE, key).value
    except NotFound as error:
        raise RuntimeError(
            f"secret {SECRET_SCOPE}/{key} não existe: "
            f"databricks secrets put-secret {SECRET_SCOPE} {key} --string-value ..."
        ) from error
    return base64.b64decode(value).decode()


@job_mutator
def failure_email(bundle: Bundle, job: Job) -> Job:
    """Avisa `alert_email` quando um job de ML falha.

    Args:
        bundle: Bundle em deploy (não usado).
        job: Job declarado no bundle (YAML ou Python).

    Returns:
        O job com `email_notifications.on_failure`.
    """
    if job.name in NO_FAILURE_EMAIL:
        return job
    return replace(job, email_notifications=JobEmailNotifications(on_failure=[secret("alert_email")]))
