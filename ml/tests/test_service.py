from datetime import datetime

import pandas as pd
import pytest

from credit_score.features import INPUT_COLUMNS, build_features
from credit_score.service import CALL_LOG_SCHEMA, build_request, call_record, status_code_of

CUSTOMER = {
    "cliente_id": 8841,
    "segmento": "varejo",
    "renda_mensal": 4200.0,
    "uso_limite": 0.8,
    "atraso_medio_12m": 3.0,
    "consultas_bureau_90d": 4,
}


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


def test_call_record_of_an_accepted_call_has_inputs_and_prediction():
    row = build_request(CUSTOMER, {"valor_solicitado": 15000.0, "prazo_meses": 24}, requests_last_24h=1)
    rec = call_record(
        row,
        request_time=datetime(2026, 11, 26, 21, 0),
        status_code=200,
        latency_ms=42.5,
        model_version="3",
        prediction={"score": 0.12, "faixa": "C", "motivos": "uso do limite alto"},
    )
    schema_columns = [c.split()[0] for c in CALL_LOG_SCHEMA.split(", ")]
    assert list(rec) == schema_columns
    assert rec["status_code"] == 200 and rec["faixa"] == "C" and rec["erro"] is None
    assert rec["prazo_meses"] == 24 and isinstance(rec["prazo_meses"], int)
    assert set(INPUT_COLUMNS) <= {k for k, v in rec.items() if v is not None}


def test_call_record_of_a_refused_call_keeps_what_was_sent():
    rec = call_record(
        {"segmento": "varejo", "uso_limite": 7},
        request_time=datetime(2026, 11, 26),
        status_code=400,
        latency_ms=10,
        model_version="3",
        error="fora do contrato: uso_limite_valido",
    )
    assert rec["uso_limite"] == 7.0 and rec["renda_mensal"] is None
    assert rec["score"] is None and rec["erro"].startswith("fora do contrato")


@pytest.mark.parametrize(
    "message, code",
    [
        ("400 Client Error: Bad Request", 400),
        ("INVALID_PARAMETER_VALUE: 1 de 1 propostas fora do contrato", 400),
        ("503 Service Unavailable", 500),
    ],
)
def test_status_code_of_endpoint_errors(message, code):
    assert status_code_of(Exception(message)) == code
