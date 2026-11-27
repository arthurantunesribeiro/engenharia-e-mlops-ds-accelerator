"""Contrato de entrada: regras de domínio aplicadas no treino, no batch e no endpoint."""

from __future__ import annotations

import pandas as pd

SEGMENTS = ["varejo", "alta_renda", "consignado"]

INPUT_CONTRACT = {
    "renda_mensal": {"gt": 0},
    "valor_solicitado": {"gt": 0},
    "prazo_meses": {"between": (1, 120)},
    "uso_limite": {"between": (0, 1)},
    "atraso_medio_12m": {"min": 0},
    "consultas_bureau_90d": {"min": 0},
    "pedidos_ultimas_24h": {"min": 0},
    "segmento": {"values": SEGMENTS},
}


def contract_sql_rules() -> dict[str, str]:
    """Gera uma expressão SQL por coluna, verdadeira quando o valor respeita o contrato.

    Returns:
        Dicionário `{nome_da_regra: expressão_sql}`, com regras no formato `<coluna>_valid`.
    """
    rules = {}
    for col, rule in INPUT_CONTRACT.items():
        parts = [f"{col} IS NOT NULL"]
        if "gt" in rule:
            parts.append(f"{col} > {rule['gt']}")
        if "min" in rule:
            parts.append(f"{col} >= {rule['min']}")
        if "between" in rule:
            parts.append(f"{col} BETWEEN {rule['between'][0]} AND {rule['between'][1]}")
        if "values" in rule:
            parts.append(f"{col} IN ({', '.join(repr(v) for v in rule['values'])})")
        rules[f"{col}_valid"] = " AND ".join(parts)
    return rules


def validate_inputs(df: pd.DataFrame) -> pd.Series:
    """Aplica o contrato linha a linha.

    Args:
        df: Entradas brutas.

    Returns:
        Série com, para cada linha, a lista de regras que falharam (lista vazia = linha válida).
    """
    failures = pd.Series([[] for _ in range(len(df))], index=df.index, dtype=object)
    for col, rule in INPUT_CONTRACT.items():
        if col not in df.columns:
            ok = pd.Series(False, index=df.index)
        elif "values" in rule:
            ok = df[col].isin(rule["values"])
        else:
            v = pd.to_numeric(df[col], errors="coerce")
            ok = v.notna()
            if "gt" in rule:
                ok &= v > rule["gt"]
            if "min" in rule:
                ok &= v >= rule["min"]
            if "between" in rule:
                ok &= v.between(*rule["between"])
        for i in df.index[~ok.fillna(False).astype(bool)]:
            failures[i] = failures[i] + [f"{col}_valid"]
    return failures
