"""Monta a tabela que o data profiling perfila: uma linha por predição, do endpoint e do batch.

- Endpoint: `chamadas_endpoint`, onde o serviço registra cada chamada já com as entradas, o score e a versão
  servida (`credit_score.service.call_record`). Só as aceitas (HTTP 200) entram.
- Batch: `scores_credito`, do scoring batch.

O perfil sobre ela é declarado em `resources/inference_profile.py`.
"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from credit_score.monitoring import PROFILE_COLUMN_TYPES, profile_columns

ML_SCHEMA = "main.credit_ml"
INFERENCE_LOG = f"{ML_SCHEMA}.credit_score_inference_log"
COLUMNS = list(PROFILE_COLUMN_TYPES)   # com os tipos da baseline: o data profiling exige o mesmo tipo nas duas


def endpoint_predictions(calls: DataFrame) -> DataFrame:
    """Coloca as chamadas aceitas pelo endpoint nas colunas da tabela de inferência.

    Args:
        calls: `chamadas_endpoint`.

    Returns:
        DataFrame nas colunas de `COLUMNS`.
    """
    return calls.filter("status_code = 200").withColumn("origem", F.lit("endpoint")).selectExpr(*profile_columns(COLUMNS))


def batch_predictions(scores: DataFrame) -> DataFrame:
    """Coloca os scores do batch nas colunas da tabela de inferência.

    Args:
        scores: `scores_credito`.

    Returns:
        DataFrame nas colunas de `COLUMNS`.
    """
    return (scores.withColumn("request_time", F.col("pontuado_em"))
            .withColumn("versao_modelo", F.col("versao_modelo").cast("string"))
            .withColumn("origem", F.lit("batch"))
            .selectExpr(*profile_columns(COLUMNS)))


def main() -> None:
    """Grava `credit_score_inference_log` com as predições do endpoint e do batch."""
    spark = SparkSession.builder.getOrCreate()
    endpoint = endpoint_predictions(spark.table(f"{ML_SCHEMA}.chamadas_endpoint"))
    batch = batch_predictions(spark.table(f"{ML_SCHEMA}.scores_credito"))
    (endpoint.unionByName(batch)
     .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(INFERENCE_LOG))


if __name__ == "__main__":
    main()
