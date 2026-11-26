"""Monta a tabela que o data profiling perfila: uma linha por predição, do endpoint e do batch.

- Endpoint: a tabela de inferência do AI Gateway (`credit_score_payload`) guarda cada chamada como JSON; aqui
  ela vira uma linha por proposta. A versão vem do `served_entity_id`, por `system.serving.served_entities`.
- Batch: `scores_credito`, do scoring batch.

O perfil sobre ela é declarado em `resources/inference_profile.py`.
"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

ML_SCHEMA = "main.credit_ml"
INFERENCE_LOG = f"{ML_SCHEMA}.credit_score_inference_log"
COLUMNS = ["request_time", "versao_modelo", "origem", "segmento", "renda_mensal", "valor_solicitado", "prazo_meses",
           "uso_limite", "atraso_medio_12m", "consultas_bureau_90d", "pedidos_ultimas_24h", "score", "faixa"]
REQUEST_SCHEMA = ("struct<dataframe_records: array<struct<segmento string, renda_mensal double, uso_limite double, "
                  "atraso_medio_12m double, consultas_bureau_90d bigint, valor_solicitado double, prazo_meses bigint, "
                  "pedidos_ultimas_24h bigint>>>")
RESPONSE_SCHEMA = "struct<predictions: array<struct<score double, faixa string, motivos string>>>"


def endpoint_predictions(payload: DataFrame, served_entities: DataFrame) -> DataFrame:
    """Desempacota as chamadas aceitas pelo endpoint em uma linha por proposta.

    Args:
        payload: Tabela de inferência do AI Gateway.
        served_entities: `system.serving.served_entities`, para traduzir o `served_entity_id` em versão.

    Returns:
        DataFrame nas colunas de `COLUMNS`.
    """
    versions = served_entities.select(
        "served_entity_id", F.col("entity_version").cast("string").alias("versao_modelo")).distinct()
    return (payload.filter("status_code = 200")
            .join(versions, "served_entity_id", "left")
            .withColumn("req", F.from_json("request", REQUEST_SCHEMA))
            .withColumn("resp", F.from_json("response", RESPONSE_SCHEMA))
            .withColumn("pair", F.explode(F.arrays_zip("req.dataframe_records", "resp.predictions")))
            .select("request_time", "versao_modelo", F.lit("endpoint").alias("origem"),
                    "pair.dataframe_records.*", "pair.predictions.score", "pair.predictions.faixa")
            .select(*COLUMNS))


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
            .select(*COLUMNS))


def main() -> None:
    """Grava `credit_score_inference_log` com as predições do endpoint e do batch."""
    spark = SparkSession.builder.getOrCreate()
    endpoint = endpoint_predictions(spark.table(f"{ML_SCHEMA}.credit_score_payload"),
                                    spark.table("system.serving.served_entities"))
    batch = batch_predictions(spark.table(f"{ML_SCHEMA}.scores_credito"))
    (endpoint.unionByName(batch)
     .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(INFERENCE_LOG))


if __name__ == "__main__":
    main()
