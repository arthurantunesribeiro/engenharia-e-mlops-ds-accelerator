# Databricks notebook source
# MAGIC %md
# MAGIC # Acompanhando um contrato da landing até a gold
# MAGIC
# MAGIC Um único contrato, camada por camada, para ver o que o pipeline (`data/src/ingestion/credit_pipeline.sql`)
# MAGIC faz com ele: como chegou, como foi guardado, limpo, corrigido e, no fim, o que virou na gold.
# MAGIC
# MAGIC | Camada | Tabela | O que acontece com o contrato |
# MAGIC | --- | --- | --- |
# MAGIC | Landing | arquivos CSV e JSON | chega sujo: valor `R$ 1.290,00`, datas em dois formatos, parcela repetida, eventos fora de ordem |
# MAGIC | Bronze | `bronze_*` | guardado como chegou, com o arquivo de origem e a hora da carga |
# MAGIC | Silver | `silver_parcelas` | tipado, CPF vira hash, duplicada sai |
# MAGIC | Silver | `silver_contratos` | CDC: o estado atual pela ordem do fato, não da chegada (SCD 1) |
# MAGIC | Silver | `silver_renda` | a renda que valia em cada período (SCD 2) |
# MAGIC | Gold | `dim_cliente`, `fato_parcelas` | esquema estrela: atraso calculado por parcela |
# MAGIC | Gold | `gold_inadimplencia_safra`, `gold_abt_credito` | o indicador do negócio e a linha que treina o modelo |
# MAGIC
# MAGIC Rode depois da primeira atualização do pipeline (job `credit_ingestion`, disparado pela chegada dos arquivos).

# COMMAND ----------

from pyspark.sql import functions as F

LANDING = "/Volumes/main/credit/landing"


dbutils.widgets.text("contrato_id", "", "Contrato (vazio = escolha automática)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0. Qual contrato acompanhar
# MAGIC Cada contrato da ABT ganha um ponto por caso que ele mostra; fica o de mais pontos.

# COMMAND ----------

candidates = spark.sql("""
WITH cdc AS (
  SELECT contrato_id, max(CASE WHEN ordem_chegada <> ordem_fato THEN 1 ELSE 0 END) AS cdc_fora_de_ordem
  FROM (SELECT contrato_id,
               row_number() OVER (PARTITION BY contrato_id ORDER BY recebido_em) AS ordem_chegada,
               row_number() OVER (PARTITION BY contrato_id ORDER BY evento_em)   AS ordem_fato
        FROM main.credit.bronze_contratos_cdc)
  GROUP BY contrato_id
),
parcelas AS (
  SELECT try_cast(contrato_id AS BIGINT) AS contrato_id,
         max(CASE WHEN valor LIKE 'R$%' THEN 1 ELSE 0 END)                       AS valor_brasileiro,
         CASE WHEN count(*) > count(DISTINCT parcela_id) THEN 1 ELSE 0 END       AS parcela_duplicada
  FROM main.credit.bronze_parcelas
  GROUP BY 1
),
renda AS (
  SELECT a.contrato_id, CASE WHEN max(r.__START_AT) > max(a.data_ref) THEN 1 ELSE 0 END AS renda_revista_depois
  FROM main.credit.gold_abt_credito a JOIN main.credit.silver_renda r USING (cliente_id)
  GROUP BY a.contrato_id
)
SELECT a.contrato_id, a.cliente_id, c.cdc_fora_de_ordem, p.valor_brasileiro, p.parcela_duplicada, r.renda_revista_depois,
       c.cdc_fora_de_ordem + p.valor_brasileiro + p.parcela_duplicada + r.renda_revista_depois AS casos
FROM main.credit.gold_abt_credito a
JOIN cdc c USING (contrato_id) JOIN parcelas p USING (contrato_id) JOIN renda r USING (contrato_id)
ORDER BY casos DESC, a.contrato_id
""")
chosen = dbutils.widgets.get("contrato_id")
row = candidates.filter(f"contrato_id = {chosen}").first() if chosen else candidates.first()
CONTRACT, CUSTOMER = row["contrato_id"], row["cliente_id"]
print(
    f"Contrato {CONTRACT} (cliente {CUSTOMER}): "
    + ", ".join(
        k for k in ("cdc_fora_de_ordem", "valor_brasileiro", "parcela_duplicada", "renda_revista_depois") if row[k]
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Landing: como os sistemas entregaram
# MAGIC Texto puro. O mesmo campo `valor` vem em dois formatos, e o CPF vem aberto.

# COMMAND ----------

landing_installments = (
    spark.read.option("header", True)
    .csv(f"{LANDING}/parcelas")
    .filter(F.col("contrato_id") == str(CONTRACT))
    .orderBy(F.col("numero").cast("int"))
)
display(landing_installments)
display(
    spark.read.json(f"{LANDING}/contratos_cdc")
    .filter(F.col("contrato_id") == CONTRACT)
    .select("status", "operacao", "evento_em", "recebido_em")
    .orderBy("recebido_em")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Bronze: guardar como chegou
# MAGIC As mesmas linhas, ainda texto, com duas colunas de rastreio: de qual arquivo vieram e quando
# MAGIC entraram. A parcela repetida continua aqui: a bronze não corrige nada.

# COMMAND ----------

bronze = spark.table("main.credit.bronze_parcelas").filter(F.col("contrato_id") == str(CONTRACT))
display(
    bronze.select("parcela_id", "numero", "valor", "vencimento", "pago_em", "cpf", "_arquivo", "_carga_em").orderBy(
        F.col("numero").cast("int")
    )
)
display(bronze.groupBy("parcela_id").count().filter("count > 1"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Silver de parcelas: tipar, proteger e deduplicar
# MAGIC `R$ 1.290,00` e `1290.00` viram o mesmo decimal, as duas formas de data viram `DATE`, o CPF vira
# MAGIC hash (LGPD) e cada parcela aparece uma vez.

# COMMAND ----------

silver = spark.table("main.credit.silver_parcelas").filter(F.col("contrato_id") == CONTRACT)
display(silver.orderBy("numero"))
print(f"bronze: {bronze.count()} linhas · silver: {silver.count()} linhas")

# COMMAND ----------

# MAGIC %md
# MAGIC Este contrato passou nas regras de qualidade. O que não passou, de outros contratos, está na
# MAGIC quarentena com o motivo: nada some sem deixar rastro.

# COMMAND ----------

display(spark.table("main.credit.quarentena_parcelas").groupBy("regras_violadas").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Silver de contratos: CDC pela ordem do fato
# MAGIC Na chegada, o evento "em atraso" veio **depois** do "renegociado" (ficou 150 dias preso num
# MAGIC sistema). Aplicando na ordem de chegada, o contrato terminaria "em atraso". Com
# MAGIC `SEQUENCE BY evento_em`, vale a ordem em que os fatos aconteceram.

# COMMAND ----------

display(
    spark.table("main.credit.bronze_contratos_cdc")
    .filter(F.col("contrato_id") == CONTRACT)
    .select("status", "operacao", "evento_em", "recebido_em")
    .orderBy("recebido_em")
)
display(
    spark.table("main.credit.silver_contratos")
    .filter(F.col("contrato_id") == CONTRACT)
    .select("contrato_id", "status", "produto", "valor", "prazo_meses", "data_concessao")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Silver de renda: a renda que valia em cada período (SCD 2)
# MAGIC A renda do cliente foi revista depois da concessão. A silver guarda as duas versões, cada uma
# MAGIC com o período em que valeu.

# COMMAND ----------

display(
    spark.table("main.credit.silver_renda")
    .filter(F.col("cliente_id") == CUSTOMER)
    .select("cliente_id", "renda_mensal", "__START_AT", "__END_AT")
    .orderBy("__START_AT")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Gold: esquema estrela
# MAGIC A dimensão traz o cliente com a renda **atual** e o CPF só em hash; o fato traz chaves e
# MAGIC medidas, com os dias de atraso de cada parcela.

# COMMAND ----------

display(spark.table("main.credit.dim_cliente").filter(F.col("cliente_id") == CUSTOMER))
display(
    spark.table("main.credit.fato_parcelas")
    .filter(F.col("contrato_id") == CONTRACT)
    .select("parcela_id", "numero", "data_vencimento", "valor", "pago_em", "dias_atraso", "atraso_90")
    .orderBy("numero")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7. Gold: o indicador do negócio e a linha do modelo
# MAGIC O contrato entra na inadimplência da sua safra. Na ABT, a renda é a que valia **na concessão**
# MAGIC (point-in-time, pelo SCD 2), não a atual da dimensão: usar a atual vazaria o futuro para o
# MAGIC treino. O alvo `inadimplente_90d` olha para depois da concessão.

# COMMAND ----------

abt_row = spark.table("main.credit.gold_abt_credito").filter(F.col("contrato_id") == CONTRACT)
display(abt_row)
vintage = abt_row.select(F.date_format("data_ref", "yyyy-MM")).first()[0]
display(spark.table("main.credit.gold_inadimplencia_safra").filter(F.col("safra") == vintage))
print(
    "renda na concessão (ABT):",
    abt_row.first()["renda_mensal"],
    "· renda atual (dim_cliente):",
    spark.table("main.credit.dim_cliente").filter(F.col("cliente_id") == CUSTOMER).first()["renda_mensal"],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8. Linhagem
# MAGIC O mesmo caminho, desenhado pelo Unity Catalog: **Catalog** › `main.credit.gold_abt_credito` ›
# MAGIC **Lineage** › **See lineage graph**.
