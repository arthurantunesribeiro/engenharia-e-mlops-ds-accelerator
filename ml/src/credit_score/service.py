"""Serviço de crédito: junta histórico, proposta e eventos recentes numa requisição ao endpoint."""

from __future__ import annotations

CUSTOMER_FEATURES = ["segmento", "renda_mensal", "uso_limite", "atraso_medio_12m", "consultas_bureau_90d"]
PROPOSAL_FIELDS = ["valor_solicitado", "prazo_meses"]


def build_request(customer: dict, proposal: dict, requests_last_24h: int) -> dict:
    """Monta a linha enviada ao endpoint a partir das três fontes.

    Args:
        customer: Histórico pré-calculado em batch (linha de `features_cliente`).
        proposal: O que a analista digitou (`valor_solicitado` e `prazo_meses`).
        requests_last_24h: Pedidos do cliente nas últimas 24h (evento recente).

    Returns:
        Dicionário com as colunas de entrada do modelo.

    Raises:
        ValueError: Se o cliente não tiver histórico ou a proposta estiver incompleta.
    """
    if not customer:
        raise ValueError("cliente sem features pré-calculadas: encaminhar para análise manual")
    missing = [c for c in PROPOSAL_FIELDS if proposal.get(c) is None]
    if missing:
        raise ValueError(f"proposta incompleta: {missing}")
    row = {c: customer[c] for c in CUSTOMER_FEATURES}
    row.update({c: proposal[c] for c in PROPOSAL_FIELDS})
    row["pedidos_ultimas_24h"] = int(requests_last_24h)
    return row


CALL_LOG_SCHEMA = (
    "request_time timestamp, status_code int, latencia_ms double, versao_modelo string, segmento string, "
    "renda_mensal double, valor_solicitado double, prazo_meses bigint, uso_limite double, atraso_medio_12m double, "
    "consultas_bureau_90d bigint, pedidos_ultimas_24h bigint, score double, faixa string, motivos string, erro string"
)


def status_code_of(error: Exception) -> int:
    """Traduz o erro de uma chamada ao endpoint no código HTTP que ele devolveu.

    Args:
        error: Exceção levantada pelo cliente do endpoint.

    Returns:
        400 para recusa da requisição (assinatura ou contrato de entrada); 500 para o resto.
    """
    message = str(error)
    return 400 if any(m in message for m in ("400", "BAD_REQUEST", "INVALID_PARAMETER_VALUE")) else 500


def call_record(
    request: dict,
    *,
    request_time,
    status_code: int,
    latency_ms: float,
    model_version: str,
    prediction: dict | None = None,
    error: str | None = None,
) -> dict:
    """Monta a linha que registra uma chamada ao endpoint, no formato de `CALL_LOG_SCHEMA`.

    Args:
        request: A linha enviada ao endpoint (pode faltar coluna, quando a chamada foi recusada).
        request_time: Momento da chamada.
        status_code: Código HTTP da resposta.
        latency_ms: Latência da chamada, em milissegundos.
        model_version: Versão servida pelo endpoint no momento da chamada.
        prediction: `score`, `faixa` e `motivos` devolvidos, quando a chamada foi aceita.
        error: Mensagem de erro, quando a chamada foi recusada.

    Returns:
        Dicionário com todas as colunas do registro.
    """
    known = {
        "request_time": request_time,
        "status_code": int(status_code),
        "latencia_ms": float(latency_ms),
        "versao_modelo": str(model_version),
        "erro": error,
        **(prediction or {}),
    }
    cast = {"double": float, "bigint": int, "string": str}
    record = {}
    for column, kind in (c.split() for c in CALL_LOG_SCHEMA.split(", ")):
        value = known[column] if column in known else request.get(column)
        record[column] = value if value is None or kind not in cast or column in known else cast[kind](value)
    return record
