"""Treino calibrado, métricas de crédito, score com faixa e motivos, e o wrapper MLflow do endpoint."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from credit_score.contract import validate_inputs
from credit_score.features import FEATURES, build_features

TARGET = "inadimplente_90d"
RISK_BANDS = [(0.05, "A"), (0.10, "B"), (0.20, "C"), (0.35, "D"), (1.01, "E")]
REASON_LABELS = {
    "comprometimento_renda": "parcela alta em relação à renda",
    "uso_limite": "uso do limite alto",
    "atraso_medio_12m": "histórico de atrasos",
    "consultas_bureau_90d": "muitas consultas recentes ao bureau",
    "pedidos_ultimas_24h": "vários pedidos nas últimas 24h",
    "log_renda": "renda baixa",
    "seg_consignado": "segmento consignado",
    "seg_alta_renda": "segmento alta renda",
}


def ks(y, p) -> float:
    """Calcula a estatística KS: a maior distância entre as distribuições de score de bons e maus.

    Args:
        y: Alvo observado (0 ou 1).
        p: Probabilidade prevista.

    Returns:
        KS entre 0 e 1.
    """
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}).sort_values("p")
    bad = (d["y"] == 1).cumsum() / max((d["y"] == 1).sum(), 1)
    good = (d["y"] == 0).cumsum() / max((d["y"] == 0).sum(), 1)
    return float((bad - good).abs().max())


def compute_metrics(y, p) -> dict:
    """Calcula AUC, Gini, KS e log loss.

    Args:
        y: Alvo observado (0 ou 1).
        p: Probabilidade prevista.

    Returns:
        Dicionário com `auc`, `gini`, `ks` e `log_loss`.
    """
    auc = roc_auc_score(y, p)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks(y, p),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
    }


def metrics_by_segment(df: pd.DataFrame, p, col: str = "segmento") -> pd.DataFrame:
    """Calcula as métricas por subpopulação, exigência comum na validação de modelos de crédito.

    Args:
        df: Base com o alvo e a coluna de segmento.
        p: Probabilidade prevista, alinhada a `df`.
        col: Coluna que define os segmentos.

    Returns:
        Uma linha por segmento com as duas classes presentes.
    """
    rows = []
    for seg, g in df.assign(_p=np.asarray(p)).groupby(col):
        if g[TARGET].nunique() == 2:
            rows.append({col: seg, "n": len(g), **compute_metrics(g[TARGET], g["_p"])})
    return pd.DataFrame(rows)


def band_table(y, p) -> pd.DataFrame:
    """Resume cada faixa de risco: volume, score médio e inadimplência observada.

    Com as faixas bem definidas, a taxa observada sobe de A para E e fica dentro do intervalo da faixa.

    Args:
        y: Alvo observado (0 ou 1).
        p: Probabilidade prevista.

    Returns:
        Uma linha por faixa presente: `faixa`, `limite_superior`, `n`, `participacao`, `score_medio`,
        `taxa_observada`.
    """
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, dtype=float)})
    d["faixa"] = [risk_band(v) for v in d["p"]]
    t = d.groupby("faixa").agg(n=("y", "size"), score_medio=("p", "mean"), taxa_observada=("y", "mean")).reset_index()
    t.insert(1, "limite_superior", t["faixa"].map({band: limit for limit, band in RISK_BANDS}).clip(upper=1.0))
    t.insert(3, "participacao", t["n"] / len(d))
    return t


def train(df: pd.DataFrame, algorithm: str = "logistic", **params):
    """Treina e calibra o classificador: o score é usado como probabilidade, não só para ordenar.

    Args:
        df: Base de treino com as entradas brutas e o alvo.
        algorithm: `"logistic"` ou `"gbm"`.
        **params: `C` para a logística; `learning_rate` e `max_depth` para o GBM.

    Returns:
        Classificador calibrado (isotônico), já treinado.
    """
    if algorithm == "logistic":
        base = make_pipeline(StandardScaler(), LogisticRegression(C=params.get("C", 1.0), max_iter=500))
    else:
        base = HistGradientBoostingClassifier(
            learning_rate=params.get("learning_rate", 0.05), max_depth=params.get("max_depth", 4), random_state=0
        )
    model = CalibratedClassifierCV(base, cv=3, method="isotonic")
    model.fit(build_features(df), df[TARGET])
    return model


def risk_band(p: float) -> str:
    """Converte uma probabilidade na faixa de risco (A a E).

    Args:
        p: Probabilidade de inadimplência.

    Returns:
        Letra da faixa.
    """
    return next(band for limit, band in RISK_BANDS if p < limit)


class CreditScoreModel:
    """Envolve o classificador e devolve score, faixa de risco e até 3 motivos.

    Os motivos são as features que mais empurram o risco para cima: a distância do cliente à
    mediana da base de treino, ponderada pelo coeficiente de uma logística de referência.

    Args:
        estimator: Classificador treinado com `predict_proba`.
        reference: Base de treino (entradas brutas e alvo), usada para medianas e pesos dos motivos.
    """

    def __init__(self, estimator, reference: pd.DataFrame):
        self.estimator = estimator
        xr = build_features(reference)
        self.median = xr.median()
        self.scale = xr.std().replace(0, 1)
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
        lr.fit(xr, reference[TARGET])
        self.weights = pd.Series(lr[-1].coef_[0], index=FEATURES)

    def score(self, inputs: pd.DataFrame) -> pd.DataFrame:
        """Pontua propostas.

        Args:
            inputs: Entradas brutas, uma linha por proposta.

        Returns:
            DataFrame com `score`, `faixa` e `motivos`, no mesmo índice de `inputs`.
        """
        x = build_features(inputs)
        p = self.estimator.predict_proba(x)[:, 1]
        contrib = ((x - self.median) / self.scale) * self.weights

        def reasons(row: pd.Series) -> str:
            top = [c for c in row.sort_values(ascending=False).index[:3] if row[c] > 0.1]
            return "; ".join(REASON_LABELS[c] for c in top) or "sem fator de risco relevante"

        return pd.DataFrame(
            {
                "score": np.round(p, 4),
                "faixa": [risk_band(v) for v in p],
                "motivos": contrib.apply(reasons, axis=1).values,
            },
            index=inputs.index,
        )


def model_requirements() -> list[str]:
    """Lista as dependências do modelo com a versão exata do ambiente atual.

    Sem versão, o build do endpoint instalaria a mais recente, e o joblib treinado aqui poderia não
    abrir. E `mlflow`, não `mlflow-skinny`: o Model Serving precisa do pacote completo.

    Returns:
        Lista no formato do `pip_requirements` do MLflow.
    """
    from importlib.metadata import PackageNotFoundError, version

    import joblib
    import sklearn

    try:
        mlflow_version = version("mlflow")
    except PackageNotFoundError:
        mlflow_version = version("mlflow-skinny")
    return [
        f"mlflow=={mlflow_version}",
        f"scikit-learn=={sklearn.__version__}",
        f"pandas=={pd.__version__}",
        f"numpy=={np.__version__}",
        f"joblib=={joblib.__version__}",
    ]


try:
    import mlflow.pyfunc
    from mlflow.exceptions import MlflowException

    class CreditScorePyfunc(mlflow.pyfunc.PythonModel):
        """Modelo PyFunc registrado no Unity Catalog e servido no endpoint.

        Aplica o contrato de entrada antes de pontuar: fora dele, a requisição inteira é recusada
        com `INVALID_PARAMETER_VALUE`, que o endpoint devolve como HTTP 400. Como o contrato viaja
        com o modelo, cada versão aplica o seu.
        """

        def load_context(self, context):
            """Carrega o `CreditScoreModel` salvo com joblib.

            Args:
                context: Contexto do MLflow, com o artefato `credit_score_model`.
            """
            import joblib

            self.model = joblib.load(context.artifacts["credit_score_model"])

        def predict(self, context, model_input, params=None):
            """Valida o contrato e pontua.

            Args:
                context: Contexto do MLflow.
                model_input: Propostas (DataFrame ou registros).
                params: Não usado.

            Returns:
                DataFrame com `score`, `faixa` e `motivos`.

            Raises:
                MlflowException: Se alguma proposta estiver fora do contrato de entrada.
            """
            inputs = pd.DataFrame(model_input)
            failures = validate_inputs(inputs)
            bad = failures[failures.str.len() > 0]
            if len(bad):
                detail = "; ".join(f"linha {i}: {', '.join(f)}" for i, f in bad.head(5).items())
                raise MlflowException.invalid_parameter_value(
                    f"{len(bad)} de {len(inputs)} propostas fora do contrato de entrada ({detail})"
                )
            return self.model.score(inputs)
except ImportError:  # pragma: no cover
    pass
