import pytest

from credit_score.features import INPUT_COLUMNS, build_features
from credit_score.service import build_request
import pandas as pd

CUSTOMER = {"cliente_id": 8841, "segmento": "varejo", "renda_mensal": 4200.0, "uso_limite": 0.8,
            "atraso_medio_12m": 3.0, "consultas_bureau_90d": 4}


def test_joins_three_sources_into_model_inputs():
    row = build_request(CUSTOMER, {"valor_solicitado": 15000.0, "prazo_meses": 24}, requests_last_24h=1)
    assert set(INPUT_COLUMNS) <= set(row)
    assert not build_features(pd.DataFrame([row])).isna().any().any()


def test_customer_without_history_goes_to_manual_review():
    with pytest.raises(ValueError, match="análise manual"):
        build_request({}, {"valor_solicitado": 1000.0, "prazo_meses": 12}, 0)


def test_incomplete_proposal_is_rejected():
    with pytest.raises(ValueError, match="incompleta"):
        build_request(CUSTOMER, {"valor_solicitado": 1000.0}, 0)
