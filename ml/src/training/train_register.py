"""Treino: busca rastreada, validação pré-deploy, registro no Unity Catalog e baseline do monitoramento.

1. A ABT passa pelo contrato de entrada, o mesmo que o endpoint aplica.
2. Uma run pai com várias filhas; escolha pela log loss, porque o score é usado como probabilidade.
3. Validação por segmento e portão de qualidade (AUC mínima).
4. Log do PyFunc com dependências de versão exata e validação pré-deploy do pacote.
5. Registro com express deployment e o alias `challenger`. A versão nova dispara o deployment job
   (resources/deployment.yml): avaliação contra o champion, aprovação na UI e deploy.
6. Baseline do data profiling: a validação pontuada pela versão registrada.
"""
import json
import os
import tempfile

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import convert_input_example_to_serving_input, infer_signature, validate_serving_input
from pyspark.sql import SparkSession

# O ambiente do modelo vem só do pip_requirements. Sem isto, o MLflow acharia o uv.lock do projeto (que o
# bundle publica junto com o código) e o copiaria para o modelo, e o endpoint seria montado a partir dele.
os.environ["MLFLOW_UV_AUTO_DETECT"] = "false"

import credit_score
from credit_score.contract import INPUT_CONTRACT
from credit_score.evaluation import split_in_time
from credit_score.features import INPUT_COLUMNS, build_features
from credit_score.monitoring import profile_columns
from credit_score.model import (TARGET, CreditScoreModel, CreditScorePyfunc, band_table, compute_metrics,
                                metrics_by_segment, model_requirements, train)

ABT_TABLE = "main.credit.gold_abt_credito"
MODEL_NAME = "main.credit_ml.credit_score"
BASELINE_TABLE = "main.credit_ml.credit_score_baseline"
MIN_AUC = 0.70
MIN_SEGMENT_AUC = 0.60   # nenhum segmento pode ficar muito abaixo do global
CANDIDATES = [
    ("logistic", {"C": 0.1}), ("logistic", {"C": 1.0}), ("logistic", {"C": 10.0}),
    ("gbm", {"learning_rate": 0.05, "max_depth": 3}), ("gbm", {"learning_rate": 0.1, "max_depth": 4}),
]


def log_evaluation(estimator, valid_df: pd.DataFrame) -> None:
    """Avalia o candidato com o avaliador de classificação do MLflow e loga tudo na run ativa.

    O `mlflow.models.evaluate` grava as métricas (`roc_auc`, `log_loss`, `precision_recall_auc`...) e
    as curvas ROC, precision-recall, de lift e de calibração. Ele só tem as probabilidades de um modelo
    no formato `mlflow.sklearn`, então o estimador é logado assim na run do candidato. Junto vai a
    tabela das faixas A–E, que é do negócio e o MLflow não tem.

    Args:
        estimator: Classificador calibrado do candidato.
        valid_df: Base de validação.
    """
    x = build_features(valid_df)
    info = mlflow.sklearn.log_model(estimator, name="estimator", serialization_format="cloudpickle",
                                    input_example=x.head(5))
    mlflow.models.evaluate(
        info.model_uri, x.assign(**{TARGET: valid_df[TARGET].to_numpy()}), targets=TARGET,
        model_type="classifier", evaluator_config={"pos_label": 1, "log_model_explainability": False})
    mlflow.log_table(band_table(valid_df[TARGET], estimator.predict_proba(x)[:, 1]), "faixas.json")


def search(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> dict:
    """Treina os candidatos em runs filhas e devolve o de menor log loss.

    Args:
        train_df: Base de treino.
        valid_df: Base de validação.

    Returns:
        O melhor candidato, com `run_id`, `name`, `estimator` e as métricas.

    Raises:
        AssertionError: Se o melhor candidato ficar abaixo da AUC mínima, no global ou em algum segmento.
    """
    results = []
    with mlflow.start_run(run_name="credit_score_search"):
        for algorithm, params in CANDIDATES:
            name = f"{algorithm}_{'_'.join(f'{k}{v}' for k, v in params.items())}"
            with mlflow.start_run(run_name=name, nested=True) as child:
                estimator = train(train_df, algorithm, **params)
                p = estimator.predict_proba(build_features(valid_df))[:, 1]
                metrics = compute_metrics(valid_df[TARGET], p)   # para a escolha e os portões
                mlflow.log_params({"algorithm": algorithm, **params})
                mlflow.log_metrics({k: metrics[k] for k in ("ks", "gini")})   # roc_auc e log_loss: evaluate
                log_evaluation(estimator, valid_df)
                by_segment = metrics_by_segment(valid_df, p)
                mlflow.log_table(by_segment, "metrics_by_segment.json")
                results.append({"run_id": child.info.run_id, "name": name, "estimator": estimator,
                                "by_segment": by_segment, **metrics})
    print(pd.DataFrame(results).drop(columns=["estimator", "by_segment"]).sort_values("log_loss").to_string(index=False))
    best = min(results, key=lambda r: r["log_loss"])
    assert best["auc"] >= MIN_AUC, f"AUC {best['auc']:.3f} abaixo do mínimo: o modelo não segue"
    seg = best["by_segment"]
    missing = set(valid_df["segmento"].unique()) - set(seg["segmento"])
    assert not missing, f"segmentos sem as duas classes na validação: {sorted(missing)}"
    weak = seg[seg["auc"] < MIN_SEGMENT_AUC]
    assert weak.empty, f"AUC abaixo de {MIN_SEGMENT_AUC} nos segmentos:\n{weak.to_string(index=False)}"
    return best


def log_model(model: CreditScoreModel, run_id: str, example: pd.DataFrame):
    """Loga o PyFunc na run escolhida, com o pacote `credit_score` e as dependências de versão exata.

    Args:
        model: Modelo com score, faixa e motivos.
        run_id: Run do candidato escolhido.
        example: Entradas de exemplo (assinatura e input example).

    Returns:
        `ModelInfo` do MLflow.
    """
    with mlflow.start_run(run_id=run_id):
        path = os.path.join(tempfile.mkdtemp(), "credit_score_model.joblib")
        joblib.dump(model, path)
        mlflow.log_dict(INPUT_CONTRACT, "input_contract.json")
        return mlflow.pyfunc.log_model(
            name="model",
            python_model=CreditScorePyfunc(),
            artifacts={"credit_score_model": path},
            code_paths=[os.path.dirname(credit_score.__file__)],   # o pacote instalado, não um caminho relativo
            signature=infer_signature(example, model.score(example)),
            input_example=example,
            pip_requirements=model_requirements(),
        )


def validate_package(model_uri: str, example: pd.DataFrame) -> None:
    """Passa o pacote pelo mesmo caminho do endpoint: o caso válido responde, o fora do contrato é recusado.

    Args:
        model_uri: URI do modelo logado.
        example: Entradas válidas.

    Raises:
        AssertionError: Se o pacote aceitar uma proposta fora do contrato.
    """
    print(validate_serving_input(model_uri, convert_input_example_to_serving_input(example)))
    bad = {**json.loads(example.head(1).to_json(orient="records"))[0], "uso_limite": 7.0}
    try:
        validate_serving_input(model_uri, json.dumps({"dataframe_records": [bad]}))
    except mlflow.exceptions.MlflowException as error:
        assert "fora do contrato" in str(error), error
        return
    raise AssertionError("o pacote aceitou uma proposta fora do contrato: não registrar")


def register(model_uri: str) -> str:
    """Registra com express deployment e marca a versão como `challenger`.

    Só o deployment job, depois da aprovação, move o `champion` (inclusive na primeira versão).

    Args:
        model_uri: URI do modelo validado.

    Returns:
        Versão registrada.
    """
    version = mlflow.register_model(model_uri, MODEL_NAME, env_pack="databricks_model_serving").version
    mlflow.MlflowClient().set_registered_model_alias(MODEL_NAME, "challenger", version)
    return version


def main() -> None:
    """Executa o treino de ponta a ponta."""
    spark = SparkSession.builder.getOrCreate()
    mlflow.set_registry_uri("databricks-uc")
    user = spark.sql("SELECT current_user()").first()[0]
    mlflow.set_experiment(f"/Users/{user}/credit_score")   # resources/foundation.yml

    train_df, valid_df = split_in_time(spark.table(ABT_TABLE).toPandas())
    best = search(train_df, valid_df)
    print(f"Escolhido: {best['name']} · log loss {best['log_loss']:.4f} · AUC {best['auc']:.3f} · KS {best['ks']:.3f}")

    model = CreditScoreModel(best["estimator"], train_df)
    example = valid_df[INPUT_COLUMNS].head(5)
    info = log_model(model, best["run_id"], example)
    validate_package(info.model_uri, example)
    version = register(info.model_uri)
    print(f"{MODEL_NAME} v{version} registrada como challenger: o deployment job avalia e aguarda aprovação")

    baseline = valid_df[INPUT_COLUMNS].join(model.score(valid_df[INPUT_COLUMNS])[["score", "faixa"]])
    (spark.createDataFrame(baseline.assign(versao_modelo=str(version)))
     .selectExpr(*profile_columns([*INPUT_COLUMNS, "score", "faixa", "versao_modelo"]))   # tipos do data profiling
     .write.mode("append").option("mergeSchema", "true").saveAsTable(BASELINE_TABLE))


if __name__ == "__main__":
    main()
