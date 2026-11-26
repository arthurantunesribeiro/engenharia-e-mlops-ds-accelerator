"""Testes do código do modelo. A qualidade do modelo (AUC global e por segmento) é validada no treino,
sobre a ABT real: ver `src/training/train_register.py`."""
import math

import pandas as pd
import pytest

from credit_score.model import RISK_BANDS, compute_metrics, ks, metrics_by_segment, risk_band


def test_output_has_score_band_and_reasons(scored_model):
    model, test_df = scored_model
    out = model.score(test_df.head(20))
    assert list(out.columns) == ["score", "faixa", "motivos"]
    assert out.index.equals(test_df.head(20).index)
    assert out["score"].between(0, 1).all()
    assert set(out["faixa"]) <= set("ABCDE")
    assert out["motivos"].str.len().gt(0).all()


def test_risk_bands_are_ordered():
    assert [risk_band(p) for p in (0.01, 0.07, 0.15, 0.3, 0.9)] == list("ABCDE")


@pytest.mark.parametrize("limit, below, at", [(limit, band, nxt) for (limit, band), (_, nxt)
                                              in zip(RISK_BANDS, RISK_BANDS[1:])])
def test_band_limit_is_exclusive(limit, below, at):
    assert risk_band(limit - 1e-9) == below
    assert risk_band(limit) == at


def test_band_covers_probability_one():
    assert risk_band(1.0) == "E"


def test_metrics_on_perfect_separation():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["auc"] == 1.0 and m["gini"] == 1.0 and m["ks"] == 1.0
    assert m["log_loss"] < 0.25


def test_metrics_on_inverted_score():
    m = compute_metrics([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
    assert m["auc"] == 0.0 and m["gini"] == -1.0


def test_log_loss_survives_extreme_probabilities():
    m = compute_metrics([0, 1], [1.0, 0.0])
    assert m["log_loss"] == pytest.approx(-math.log(1e-6))


def test_ks_measures_max_distance_between_good_and_bad():
    assert ks([0, 1], [0.3, 0.7]) == 1.0
    assert ks([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(0.5)


def test_metrics_by_segment_skips_segments_with_one_class():
    df = pd.DataFrame({"segmento": ["a", "a", "a", "a", "b", "b"], "inadimplente_90d": [0, 0, 1, 1, 0, 0]})
    out = metrics_by_segment(df, [0.1, 0.2, 0.8, 0.9, 0.5, 0.5])
    assert out["segmento"].tolist() == ["a"]
    assert out.loc[0, "n"] == 4 and out.loc[0, "auc"] == 1.0
