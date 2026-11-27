"""Monitoramento: classificação do drift calculado pelo data profiling e alertas."""
from __future__ import annotations

import pandas as pd

OUT_OF_CONTRACT_LIMIT = 0.02  # acima de 2% de chamadas recusadas ou propostas em quarentena, alguém olha

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


def classify_psi(value: float) -> str:
    """Classifica o PSI com a régua do data profiling do Databricks.

    Args:
        value: PSI de uma coluna (`population_stability_index` da tabela de drift).

    Returns:
        `"estável"` (< 0,1), `"atenção"` (< 0,2) ou `"investigar"`.
    """
    if value < 0.1:
        return "estável"
    return "atenção" if value < 0.2 else "investigar"


def build_alerts(drift: pd.DataFrame, out_of_contract_rate: float = 0.0) -> list[str]:
    """Transforma o drift e a qualidade da entrada em mensagens de alerta, com o mais grave primeiro.

    Dado fora do contrato vem antes do drift: se a entrada quebrou, o PSI mede o defeito, não o mundo.

    Args:
        drift: Uma linha por coluna e segmento, com `column_name`, `slice_value` (nulo = base inteira) e `psi`.
        out_of_contract_rate: Fração do tráfego recusada ou em quarentena.

    Returns:
        Lista de mensagens (vazia se tudo estável).
    """
    messages = []
    if out_of_contract_rate > OUT_OF_CONTRACT_LIMIT:
        messages.append(f"{out_of_contract_rate:.1%} do tráfego fora do contrato de entrada "
                        f"(limite {OUT_OF_CONTRACT_LIMIT:.0%}): investigar a origem antes do drift")
    unstable = drift[drift["psi"].map(classify_psi) != "estável"].sort_values("psi", ascending=False)
    for _, r in unstable.iterrows():
        scope = "base inteira" if pd.isna(r["slice_value"]) else f"segmento {r['slice_value']}"
        messages.append(f"PSI de {r['column_name']} ({scope}) = {r['psi']:.3f} ({classify_psi(r['psi'])})")
    return messages
