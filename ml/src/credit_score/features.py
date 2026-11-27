"""Transformação das entradas brutas nas features do modelo: uma função só, em todos os caminhos."""

from __future__ import annotations

import numpy as np
import pandas as pd

INPUT_COLUMNS = [
    "renda_mensal",
    "valor_solicitado",
    "prazo_meses",
    "uso_limite",
    "atraso_medio_12m",
    "consultas_bureau_90d",
    "pedidos_ultimas_24h",
    "segmento",
]
FEATURES = [
    "comprometimento_renda",
    "uso_limite",
    "atraso_medio_12m",
    "consultas_bureau_90d",
    "pedidos_ultimas_24h",
    "log_renda",
    "seg_consignado",
    "seg_alta_renda",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula as features do modelo a partir das entradas brutas.

    Args:
        df: Entradas brutas, com as colunas de `INPUT_COLUMNS` (renda MENSAL).

    Returns:
        DataFrame com as colunas de `FEATURES`, sem nulos e com valores limitados.

    Raises:
        ValueError: Se faltar alguma coluna de `INPUT_COLUMNS`.
    """
    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"colunas ausentes: {missing}")
    x = pd.DataFrame(index=df.index)
    income = df["renda_mensal"].astype(float).fillna(1.0).clip(lower=1.0)
    installment = df["valor_solicitado"].astype(float).fillna(0) / df["prazo_meses"].astype(float).fillna(1).clip(
        lower=1
    )
    x["comprometimento_renda"] = (installment / income).clip(0, 3)
    x["uso_limite"] = df["uso_limite"].astype(float).fillna(0.5).clip(0, 1)
    x["atraso_medio_12m"] = df["atraso_medio_12m"].astype(float).fillna(0).clip(lower=0)
    x["consultas_bureau_90d"] = df["consultas_bureau_90d"].astype(float).fillna(0).clip(lower=0)
    x["pedidos_ultimas_24h"] = df["pedidos_ultimas_24h"].astype(float).fillna(0).clip(lower=0)
    x["log_renda"] = np.log(income)
    x["seg_consignado"] = (df["segmento"] == "consignado").astype(float)
    x["seg_alta_renda"] = (df["segmento"] == "alta_renda").astype(float)
    return x[FEATURES]
