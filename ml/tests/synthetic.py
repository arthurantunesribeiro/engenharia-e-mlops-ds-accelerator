"""Dados sintéticos no formato da ABT de crédito, para os testes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from credit_score.contract import SEGMENTS

CHANNELS = ["app", "agencia", "correspondente"]


def generate_abt(n: int = 20_000, start: str = "2025-01-01", months: int = 12, seed: int = 42,
                 income_shift: float = 1.0, stress: float = 0.0) -> pd.DataFrame:
    """Gera contratos com features na concessão e o alvo `inadimplente_90d`.

    Args:
        n: Número de contratos.
        start: Data inicial das concessões.
        months: Janela de concessões, em meses.
        seed: Semente do gerador aleatório.
        income_shift: Multiplicador da renda; abaixo de 1 simula clientes com renda menor que a do treino.
        stress: Acima de 0 simula mais uso do limite e mais consultas ao bureau, como numa piora econômica.

    Returns:
        DataFrame com uma linha por contrato, nas colunas da ABT.
    """
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(start) + pd.to_timedelta(rng.integers(0, months * 30, n), unit="D")
    segment = rng.choice(SEGMENTS, n, p=[0.6, 0.15, 0.25])
    base_income = np.where(segment == "alta_renda", 15_000, np.where(segment == "consignado", 4_500, 3_500))
    income = np.round(rng.lognormal(np.log(base_income), 0.45) * income_shift, 2)
    amount = np.round(income * rng.uniform(1.5, 12, n), 2)
    term = rng.choice([6, 12, 18, 24, 36, 48], n)
    limit_usage = np.clip(rng.beta(2, 3, n) + np.where(segment == "varejo", 0.1, 0) + 0.25 * stress, 0, 1)
    avg_delay = np.round(rng.gamma(1.2, 4, n) * (1.3 - income / income.max()), 1)
    bureau_queries = rng.poisson(np.where(segment == "varejo", 2.5, 1.2) * (1 + stress))
    recent_requests = rng.poisson(0.3, n)
    channel = rng.choice(CHANNELS, n, p=[0.55, 0.25, 0.20])

    burden = np.clip((amount / term) / income, 0, 3)
    logit = (-5.1 + 2.4 * burden + 1.6 * limit_usage + 0.08 * avg_delay + 0.18 * bureau_queries
             + 0.45 * recent_requests - 0.6 * (segment == "consignado") - 0.4 * (segment == "alta_renda"))
    target = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    return pd.DataFrame({
        "contrato_id": np.arange(1, n + 1),
        "cliente_id": rng.integers(1, max(2, n // 3) + 1, n),
        "data_ref": dates.normalize(),
        "segmento": segment,
        "canal": channel,
        "renda_mensal": income,
        "valor_solicitado": amount,
        "prazo_meses": term,
        "uso_limite": np.round(limit_usage, 3),
        "atraso_medio_12m": avg_delay,
        "consultas_bureau_90d": bureau_queries,
        "pedidos_ultimas_24h": recent_requests,
        "inadimplente_90d": target,
    })
