"""Monitoramento: os tipos das colunas que o data profiling compara."""
from __future__ import annotations

# Tipos das colunas que o data profiling compara. A tabela de inferência e a baseline precisam ter o mesmo
# tipo em cada coluna em comum, venha o dado do endpoint, do batch ou do treino.
PROFILE_COLUMN_TYPES = {
    "request_time": "timestamp", "versao_modelo": "string", "origem": "string", "segmento": "string",
    "renda_mensal": "double", "valor_solicitado": "double", "prazo_meses": "int", "uso_limite": "double",
    "atraso_medio_12m": "double", "consultas_bureau_90d": "int", "pedidos_ultimas_24h": "int",
    "score": "double", "faixa": "string",
}


def profile_columns(columns: list[str]) -> list[str]:
    """Expressões SQL que convertem as colunas para os tipos do data profiling.

    Args:
        columns: Colunas, todas presentes em `PROFILE_COLUMN_TYPES`.

    Returns:
        Uma expressão `CAST(coluna AS tipo) AS coluna` por coluna, na ordem recebida.
    """
    return [f"CAST({c} AS {PROFILE_COLUMN_TYPES[c]}) AS {c}" for c in columns]


# Limites dos alertas: os mesmos dos SQL alerts (resources/model_alerts.py).
DRIFT_LIMIT = 0.2               # PSI acima disso: "investigar", na régua do data profiling
OUT_OF_CONTRACT_LIMIT = 0.02    # acima de 2% de chamadas recusadas ou propostas em quarentena, alguém olha


def chat_message(drift: dict, out_of_contract: dict) -> str | None:
    """Monta o aviso do Google Chat a partir das consultas dos SQL alerts.

    Entrada fora do contrato vem primeiro: se a entrada quebrou, o PSI mede o defeito, não o mundo.

    Args:
        drift: Linha de `resources/alerts/drift.sql` (`maior_psi`, `onde`, `colunas_acima_de_0_2`).
        out_of_contract: Linha de `resources/alerts/fora_do_contrato.sql` (`maior_taxa_fora_do_contrato`,
            `taxa_endpoint_7d`, `taxa_ultimo_lote`).

    Returns:
        O texto do aviso, ou `None` se nenhum limite foi ultrapassado.
    """
    def number(row, key):
        return None if row.get(key) is None else float(row[key])

    lines = []
    rate = number(out_of_contract, "maior_taxa_fora_do_contrato")
    if rate is not None and rate > OUT_OF_CONTRACT_LIMIT:
        endpoint, batch = number(out_of_contract, "taxa_endpoint_7d"), number(out_of_contract, "taxa_ultimo_lote")
        lines.append(f"⚠️ *Entrada fora do contrato: {rate:.1%}* (limite {OUT_OF_CONTRACT_LIMIT:.0%}) · endpoint em "
                     f"7 dias {endpoint or 0:.1%} · último lote do batch {batch or 0:.1%}")
    psi = number(drift, "maior_psi")
    if psi is not None and psi > DRIFT_LIMIT:
        lines.append(f"⚠️ *Drift: PSI {psi:.2f}* em {drift.get('onde')} (limite {DRIFT_LIMIT}) · "
                     f"{int(drift.get('colunas_acima_de_0_2') or 0)} coluna(s)/segmento(s) acima do limite")
    if not lines:
        return None
    return "\n".join(["*credit_score · monitoramento*", *lines,
                      "Próximo passo: diagnosticar no painel do perfil de inferência; retreino só por decisão."])
