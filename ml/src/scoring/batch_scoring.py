"""Scoring batch: contrato de entrada em SQL, quarentena e score com o alias champion."""

import mlflow
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from credit_score.contract import contract_sql_rules
from credit_score.features import INPUT_COLUMNS
from credit_score.settings import catalog_from_args

SOURCE_TABLE = "credit.gold_propostas_pendentes"
MAX_QUARANTINE_RATE = 0.05


def main() -> None:
    """Separa a quarentena e pontua as propostas válidas em `scores_credito`.

    Raises:
        AssertionError: Se a quarentena passar de `MAX_QUARANTINE_RATE` do lote.
    """
    catalog = catalog_from_args()
    model_name = f"{catalog}.credit_ml.credit_score"
    spark = SparkSession.builder.getOrCreate()
    spark.sql(f"USE CATALOG {catalog}")
    mlflow.set_registry_uri("databricks-uc")

    failures = F.filter(
        F.array(
            *[
                F.when(~F.coalesce(F.expr(expr), F.lit(False)), F.lit(name))
                for name, expr in contract_sql_rules().items()
            ]
        ),
        lambda x: x.isNotNull(),
    )
    proposals = spark.table(SOURCE_TABLE).withColumn("falhas", failures)
    valid = proposals.filter(F.size("falhas") == 0).drop("falhas")
    quarantine = proposals.filter(F.size("falhas") > 0).withColumn("barrado_em", F.current_timestamp())

    total, quarantined = proposals.count(), quarantine.count()
    print(f"{quarantined} de {total} propostas em quarentena ({quarantined / total:.2%})")
    quarantine.write.mode("append").option("mergeSchema", "true").saveAsTable("credit_ml.propostas_quarentena")
    assert quarantined / total <= MAX_QUARANTINE_RATE, f"{quarantined / total:.1%} do lote fora do contrato"

    predict = mlflow.pyfunc.spark_udf(
        spark, f"models:/{model_name}@champion", result_type="score double, faixa string, motivos string"
    )
    version = mlflow.MlflowClient().get_model_version_by_alias(model_name, "champion").version
    (
        valid.withColumn("out", predict(*[F.col(c) for c in INPUT_COLUMNS]))
        .select("*", "out.score", "out.faixa", "out.motivos")
        .drop("out")
        .withColumn("versao_modelo", F.lit(version))
        .withColumn("pontuado_em", F.current_timestamp())
        .write.mode("overwrite")
        .saveAsTable("credit_ml.scores_credito")
    )


if __name__ == "__main__":
    main()
