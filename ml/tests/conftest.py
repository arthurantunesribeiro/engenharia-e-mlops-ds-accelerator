"""Fixtures compartilhadas: base sintética, modelo treinado e o pacote MLflow como o treino o registra."""
import os

os.environ["MLFLOW_UV_AUTO_DETECT"] = "false"   # como em src/training/train_register.py

import joblib
import pandas as pd
import pytest

from synthetic import generate_abt
from credit_score.features import INPUT_COLUMNS
from credit_score.model import CreditScoreModel, train

PROPOSAL = {"renda_mensal": 4200.0, "valor_solicitado": 15000.0, "prazo_meses": 24, "uso_limite": 0.8,
            "atraso_medio_12m": 3.0, "consultas_bureau_90d": 4, "pedidos_ultimas_24h": 1, "segmento": "varejo"}


def proposal(**changes) -> pd.DataFrame:
    """Uma proposta válida, com os campos alterados por `changes`."""
    return pd.DataFrame([{**PROPOSAL, **changes}])


@pytest.fixture(scope="session")
def base():
    return generate_abt(n=6000, seed=1)


@pytest.fixture(scope="session")
def scored_model(base):
    train_df, test_df = base.iloc[:4500], base.iloc[4500:]
    return CreditScoreModel(train(train_df, "logistic"), train_df), test_df


@pytest.fixture(scope="session")
def model_package(tmp_path_factory):
    """O modelo empacotado como em `src/training/train_register.py`: PyFunc, assinatura, código e dependências."""
    import mlflow
    from mlflow.models import infer_signature
    import credit_score
    from credit_score.model import CreditScorePyfunc, model_requirements

    folder = tmp_path_factory.mktemp("model")
    data = generate_abt(n=4000, seed=3)
    model = CreditScoreModel(train(data), data)
    joblib.dump(model, folder / "credit_score_model.joblib")
    example = data[INPUT_COLUMNS].head(5)
    mlflow.pyfunc.save_model(
        str(folder / "package"), python_model=CreditScorePyfunc(),
        artifacts={"credit_score_model": str(folder / "credit_score_model.joblib")},
        code_paths=[os.path.dirname(credit_score.__file__)],
        signature=infer_signature(example, model.score(example)), input_example=example,
        pip_requirements=model_requirements())
    return str(folder / "package"), example
