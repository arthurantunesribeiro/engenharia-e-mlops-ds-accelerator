"""Janela de validação e relatório de um score: o mesmo no treino e no deployment job."""

from __future__ import annotations

import pandas as pd

from credit_score.contract import validate_inputs
from credit_score.model import TARGET, band_table, compute_metrics, metrics_by_segment

MAX_OUT_OF_CONTRACT = 0.01


def split_in_time(abt: pd.DataFrame, valid_share: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica o contrato e separa treino (safras antigas) e validação (safras recentes).

    Dividir aleatoriamente dados com ordem no tempo vazaria o futuro para o treino. O deployment job usa
    a mesma validação para comparar a versão nova e o champion na mesma janela.

    Args:
        abt: ABT completa.
        valid_share: Fração mais recente (por `data_ref`) que vai para a validação.

    Returns:
        Treino e validação.

    Raises:
        AssertionError: Se mais de 1% da ABT estiver fora do contrato de entrada.
    """
    out = validate_inputs(abt).str.len() > 0
    assert out.mean() < MAX_OUT_OF_CONTRACT, f"{out.mean():.1%} da ABT fora do contrato: revisar o pipeline"
    abt = abt[~out].sort_values("data_ref")
    cutoff = abt["data_ref"].quantile(1 - valid_share)
    return abt[abt["data_ref"] < cutoff], abt[abt["data_ref"] >= cutoff]


def score_report(df: pd.DataFrame, p) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Resume um score numa base com o alvo: métricas, métricas por segmento e faixas de risco.

    Args:
        df: Base com o alvo e o segmento.
        p: Probabilidade prevista, alinhada a `df`.

    Returns:
        Métricas globais, métricas por segmento e a tabela das faixas A–E.
    """
    return compute_metrics(df[TARGET], p), metrics_by_segment(df, p), band_table(df[TARGET], p)
