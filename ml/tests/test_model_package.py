"""O modelo como o endpoint o recebe: atrás do servidor de scoring do MLflow, e a validação pré-deploy."""
import json
from pathlib import Path

import pytest

from conftest import PROPOSAL


@pytest.fixture(scope="module")
def endpoint(model_package):
    import mlflow
    from fastapi.testclient import TestClient
    from mlflow.pyfunc import scoring_server
    return TestClient(scoring_server.init(mlflow.pyfunc.load_model(model_package[0])))


def call(client, *rows):
    return client.post("/invocations", json={"dataframe_records": list(rows)})


def test_endpoint_returns_score_band_and_reasons(endpoint):
    r = call(endpoint, PROPOSAL)
    assert r.status_code == 200
    out = r.json()["predictions"][0]
    assert 0 <= out["score"] <= 1 and out["faixa"] in list("ABCDE") and out["motivos"]


def test_missing_field_returns_400_from_signature(endpoint):
    assert call(endpoint, {k: v for k, v in PROPOSAL.items() if k != "renda_mensal"}).status_code == 400


@pytest.mark.parametrize("changes,rule", [({"uso_limite": 7.0}, "uso_limite_valid"),
                                          ({"segmento": "premium"}, "segmento_valid"),
                                          ({"prazo_meses": 0}, "prazo_meses_valid")])
def test_out_of_contract_returns_400_with_the_rule(endpoint, changes, rule):
    r = call(endpoint, {**PROPOSAL, **changes})
    assert r.status_code == 400 and rule in r.json()["message"]


def test_one_bad_row_rejects_the_batch(endpoint):
    r = call(endpoint, PROPOSAL, {**PROPOSAL, "segmento": "premium"})
    assert r.status_code == 400 and "1 de 2 propostas" in r.json()["message"]


def test_requirements_are_pinned_with_full_mlflow(model_package):
    lines = Path(model_package[0], "requirements.txt").read_text().split()
    assert all("==" in line for line in lines)
    assert any(line.startswith("mlflow==") for line in lines)


def test_pre_deploy_example_payload_passes(model_package):
    from mlflow.models import convert_input_example_to_serving_input, validate_serving_input
    out = validate_serving_input(model_package[0], convert_input_example_to_serving_input(model_package[1]))
    assert list(out.columns) == ["score", "faixa", "motivos"]


def test_pre_deploy_out_of_contract_payload_is_rejected(model_package):
    from mlflow.exceptions import MlflowException
    from mlflow.models import validate_serving_input
    with pytest.raises(MlflowException, match="fora do contrato"):
        validate_serving_input(model_package[0], json.dumps({"dataframe_records": [{**PROPOSAL, "prazo_meses": 0}]}))


@pytest.mark.environment
def test_package_runs_in_isolated_environment(model_package, capfd):
    """Recria do zero o ambiente do requirements.txt, como o build do endpoint. Precisa do uv."""
    import mlflow
    mlflow.models.predict(model_uri=model_package[0], input_data=model_package[1], env_manager="uv")
    assert '"faixa"' in capfd.readouterr().out
