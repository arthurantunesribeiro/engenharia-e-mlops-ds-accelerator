import pandas as pd

from credit_score.features import INPUT_COLUMNS
from credit_score.monitoring import PROFILE_COLUMN_TYPES, build_alerts, classify_psi, profile_columns

DRIFT = pd.DataFrame({"column_name": ["renda_mensal", "uso_limite", "score"],
                      "slice_value": [None, "varejo", None], "psi": [0.04, 0.31, 0.12]})


def test_psi_follows_databricks_thresholds():
    assert [classify_psi(v) for v in (0.05, 0.15, 0.25)] == ["estável", "atenção", "investigar"]


def test_drift_alerts_come_worst_first_and_skip_stable_columns():
    alerts = build_alerts(DRIFT)
    assert len(alerts) == 2
    assert alerts[0].startswith("PSI de uso_limite (segmento varejo)")
    assert "score (base inteira)" in alerts[1]


def test_out_of_contract_traffic_comes_before_drift():
    assert build_alerts(DRIFT.iloc[:1], out_of_contract_rate=0.01) == []
    assert "fora do contrato" in build_alerts(DRIFT, out_of_contract_rate=0.08)[0]


def test_profile_columns_cast_to_the_baseline_types():
    exprs = profile_columns(["prazo_meses", "score", "segmento"])
    assert exprs == ["CAST(prazo_meses AS int) AS prazo_meses", "CAST(score AS double) AS score",
                     "CAST(segmento AS string) AS segmento"]


def test_profile_types_cover_every_model_input():
    assert set(INPUT_COLUMNS) | {"score", "faixa", "versao_modelo"} <= set(PROFILE_COLUMN_TYPES)
