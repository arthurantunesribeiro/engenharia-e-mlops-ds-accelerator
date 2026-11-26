"""Treino: busca rastreada, validação pré-deploy, registro no Unity Catalog e baseline do monitoramento.

1. A ABT passa pelo contrato de entrada, o mesmo que o endpoint aplica.
2. Uma run pai com várias filhas; escolha pela log loss, porque o score é usado como probabilidade.
3. Validação por segmento e portão de qualidade (AUC mínima).
4. Log do PyFunc com dependências de versão exata e validação pré-deploy do pacote.
5. Registro com express deployment e alias: a primeira versão vira `champion`, as seguintes `challenger`.
6. Baseline do data profiling: a validação pontuada pela versão registrada.
"""
import json
import os
import tempfile

import joblib
import mlflow
import pandas as pd
from mlflow.models import convert_input_example_to_serving_input, infer_signature, validate_serving_input
from pyspark.sql import SparkSession

# O ambiente do modelo vem só do pip_requirements. Sem isto, o MLflow acharia o uv.lock do projeto (que o
# bundle publica junto com o código) e o copiaria para o modelo, e o endpoint seria montado a partir dele.
os.environ["MLFLOW_UV_AUTO_DETECT"] = "false"

import credit_score
from credit_score.contract import INPUT_CONTRACT, validate_inputs
from credit_score.features import INPUT_COLUMNS, build_features
from credit_score.model import (CreditScoreModel, CreditScorePyfunc, compute_metrics, metrics_by_segment,
                                model_requirements, train)

ABT_TABLE = "main.credit.gold_abt_credito"
MODEL_NAME = "main.credit_ml.credit_score"
BASELINE_TABLE = "main.credit_ml.credit_score_baseline"
MIN_AUC = 0.70
MIN_SEGMENT_AUC = 0.60   # nenhum segmento pode ficar muito abaixo do global
CANDIDATES = [
    ("logistic", {"C": 0.1}), ("logistic", {"C": 1.0}), ("logistic", {"C": 10.0}),
    ("gbm", {"learning_rate": 0.05, "max_depth": 3}), ("gbm", {"learning_rate": 0.1, "max_depth": 4}),
]


def split_in_time(abt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica o contrato e separa treino (safras antigas) e validação (safras recentes).

    Dividir aleatoriamente dados com ordem no tempo vazaria o futuro para o treino.

    Args:
        abt: ABT completa.

    Returns:
        Treino e validação.

    Raises:
        AssertionError: Se mais de 1% da ABT estiver fora do contrato de entrada.
    """
    out = validate_inputs(abt).str.len() > 0
    assert out.mean() < 0.01, f"{out.mean():.1%} da ABT fora do contrato: revisar o pipeline"
    abt = abt[~out].sort_values("data_ref")
    cutoff = abt["data_ref"].quantile(0.75)
    return abt[abt["data_ref"] < cutoff], abt[abt["data_ref"] >= cutoff]


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
                metrics = compute_metrics(valid_df["inadimplente_90d"], p)
                mlflow.log_params({"algorithm": algorithm, **params})
                mlflow.log_metrics(metrics)
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


def register(model_uri: str) -> tuple[str, str]:
    """Registra com express deployment e define o alias.

    Args:
        model_uri: URI do modelo validado.

    Returns:
        Versão registrada e alias atribuído.
    """
    client = mlflow.MlflowClient()
    version = mlflow.register_model(model_uri, MODEL_NAME, env_pack="databricks_model_serving").version
    try:
        client.get_model_version_by_alias(MODEL_NAME, "champion")
        alias = "challenger"
    except mlflow.exceptions.MlflowException:
        alias = "champion"
    client.set_registered_model_alias(MODEL_NAME, alias, version)
    return version, alias


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
    version, alias = register(info.model_uri)
    print(f"{MODEL_NAME} v{version} registrada com o alias {alias}")

    baseline = valid_df[INPUT_COLUMNS].join(model.score(valid_df[INPUT_COLUMNS])[["score", "faixa"]])
    (spark.createDataFrame(baseline.assign(versao_modelo=str(version)))
     .write.mode("append").option("mergeSchema", "true").saveAsTable(BASELINE_TABLE))


if __name__ == "__main__":
    main()
