import numpy as np
import pytest

from conftest import proposal
from credit_score.features import FEATURES, build_features


def test_base_case_computes_income_burden():
    x = build_features(proposal(valor_solicitado=12000.0, prazo_meses=12, renda_mensal=4000.0))
    assert list(x.columns) == FEATURES
    assert x.loc[0, "comprometimento_renda"] == pytest.approx(1000 / 4000)


def test_nulls_do_not_produce_nan():
    x = build_features(proposal(renda_mensal=None, uso_limite=None, atraso_medio_12m=None))
    assert not x.isna().any().any()


def test_extreme_values_stay_within_limits():
    x = build_features(proposal(renda_mensal=0, uso_limite=7, valor_solicitado=1e9, consultas_bureau_90d=-3))
    assert x.loc[0, "comprometimento_renda"] <= 3
    assert 0 <= x.loc[0, "uso_limite"] <= 1
    assert x.loc[0, "consultas_bureau_90d"] >= 0
    assert np.isfinite(x.values).all()


def test_missing_column_is_rejected():
    with pytest.raises(ValueError, match="colunas ausentes"):
        build_features(proposal().drop(columns=["renda_mensal"]))


def test_annual_income_by_mistake_changes_the_feature():
    """O skew clássico: renda anual no lugar da mensal derruba o comprometimento em 12x."""
    monthly = build_features(proposal())
    annual = build_features(proposal(renda_mensal=4200.0 * 12))
    assert annual.loc[0, "comprometimento_renda"] == pytest.approx(monthly.loc[0, "comprometimento_renda"] / 12)
