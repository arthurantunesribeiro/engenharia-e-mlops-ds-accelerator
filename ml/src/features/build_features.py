"""Features: a fotografia mais recente de cada cliente, a partir da ABT."""

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

ABT_TABLE = "main.credit.gold_abt_credito"
FEATURES_TABLE = "main.credit_ml.features_cliente"


def main() -> None:
    """Grava `features_cliente` com uma linha por cliente (a do contrato mais recente)."""
    spark = SparkSession.builder.getOrCreate()
    latest = Window.partitionBy("cliente_id").orderBy(F.col("data_ref").desc())
    (
        spark.table(ABT_TABLE)
        .withColumn("_n", F.row_number().over(latest))
        .filter("_n = 1")
        .select(
            "cliente_id",
            "segmento",
            "renda_mensal",
            "uso_limite",
            "atraso_medio_12m",
            "consultas_bureau_90d",
            F.col("data_ref").alias("atualizado_em"),
        )
        .write.mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(FEATURES_TABLE)
    )


if __name__ == "__main__":
    main()
