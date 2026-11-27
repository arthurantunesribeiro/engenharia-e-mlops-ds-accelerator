"""Monta a tabela de inferência perfilada: uma linha por predição, do endpoint e do batch."""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from credit_score.monitoring import PROFILE_COLUMN_TYPES, profile_columns
from credit_score.settings import catalog_from_args

INFERENCE_LOG = "credit_ml.credit_score_inference_log"
COLUMNS = list(PROFILE_COLUMN_TYPES)


def endpoint_predictions(calls: DataFrame) -> DataFrame:
    """Coloca as chamadas aceitas pelo endpoint nas colunas da tabela de inferência.

    Args:
        calls: `chamadas_endpoint`.

    Returns:
        DataFrame nas colunas de `COLUMNS`.
    """
    return (
        calls.filter("status_code = 200").withColumn("origem", F.lit("endpoint")).selectExpr(*profile_columns(COLUMNS))
    )


def batch_predictions(scores: DataFrame) -> DataFrame:
    """Coloca os scores do batch nas colunas da tabela de inferência.

    Args:
        scores: `scores_credito`.

    Returns:
        DataFrame nas colunas de `COLUMNS`.
    """
    return (
        scores.withColumn("request_time", F.col("pontuado_em"))
        .withColumn("versao_modelo", F.col("versao_modelo").cast("string"))
        .withColumn("origem", F.lit("batch"))
        .selectExpr(*profile_columns(COLUMNS))
    )


def main() -> None:
    """Grava `credit_score_inference_log` com as predições do endpoint e do batch."""
    spark = SparkSession.builder.getOrCreate()
    spark.sql(f"USE CATALOG {catalog_from_args()}")
    endpoint = endpoint_predictions(spark.table("credit_ml.chamadas_endpoint"))
    batch = batch_predictions(spark.table("credit_ml.scores_credito"))
    (endpoint.unionByName(batch).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(INFERENCE_LOG))


if __name__ == "__main__":
    main()
