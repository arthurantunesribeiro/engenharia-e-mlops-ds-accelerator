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
