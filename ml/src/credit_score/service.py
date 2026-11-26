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
