# Databricks notebook source
# MAGIC %md
# MAGIC # Simulação dos sistemas de origem
# MAGIC
# MAGIC Fora do projeto em produção: lá, os sistemas de origem entregam os arquivos. Aqui simulamos o
# MAGIC que cinco sistemas deixariam na área de pouso (`/Volumes/main/credit/landing`), de onde o
# MAGIC pipeline `data/src/ingestion/credit_pipeline.sql` lê:
# MAGIC
# MAGIC | Pasta | Formato | Problemas de propósito |
# MAGIC | --- | --- | --- |
# MAGIC | `parcelas` | CSV | `R$ 1.290,00` e `1290.00`, datas em dois formatos, duplicadas, CPF aberto, linhas sem contrato |
# MAGIC | `contratos_cdc` | JSON | eventos INSERT, UPDATE e DELETE, alguns chegando fora de ordem |
# MAGIC | `clientes` | JSON | cadastro |
# MAGIC | `renda` | JSON | a renda muda com o tempo (SCD 2) |
# MAGIC | `propostas` | JSON | o que se sabia no pedido; `lote_0002` traz propostas novas, ainda sem contrato, com algumas sujas |
# MAGIC
# MAGIC Grava uma vez só: o pipeline lê cada arquivo uma única vez, e regravar duplicaria a bronze.

# COMMAND ----------

import numpy as np
import pandas as pd

LANDING = "/Volumes/main/credit/landing"   # volume declarado em resources/foundation.yml
rng = np.random.default_rng(42)
START, END = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")
DAYS = (END - START).days
N_CUSTOMERS, N_CONTRACTS = 3000, 6000

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clientes, renda e propostas

# COMMAND ----------

segment = rng.choice(["varejo", "alta_renda", "consignado"], N_CUSTOMERS, p=[0.6, 0.15, 0.25])
base_income = np.where(segment == "alta_renda", 15000, np.where(segment == "consignado", 4500, 3500))
income0 = np.round(rng.lognormal(np.log(base_income), 0.45), 2)
hidden_risk = rng.gamma(1.0, 1.0, N_CUSTOMERS) * np.where(segment == "varejo", 1.3, 0.8)   # o "mundo real" escondido

customers = pd.DataFrame({
    "cliente_id": np.arange(1, N_CUSTOMERS + 1),
    "cpf": [f"{a:03d}.{b:03d}.{c:03d}-{d % 100:02d}" for a, b, c, d in rng.integers(0, 1000, (N_CUSTOMERS, 4))],
    "segmento": segment,
    "canal": rng.choice(["app", "agencia", "correspondente"], N_CUSTOMERS, p=[0.55, 0.25, 0.20]),
    "cadastrado_em": START - pd.to_timedelta(rng.integers(30, 720, N_CUSTOMERS), unit="D"),
})

# renda: a do cadastro e, para 35% dos clientes, uma revisão depois
revised = rng.random(N_CUSTOMERS) < 0.35
income = pd.concat([
    pd.DataFrame({"cliente_id": customers["cliente_id"], "renda_mensal": income0, "atualizado_em": customers["cadastrado_em"]}),
    pd.DataFrame({"cliente_id": customers.loc[revised, "cliente_id"].values,
                  "renda_mensal": np.round(income0[revised] * rng.uniform(0.6, 1.6, revised.sum()), 2),
                  "atualizado_em": START + pd.to_timedelta(rng.integers(0, DAYS, revised.sum()), unit="D")}),
], ignore_index=True)

# propostas aprovadas: cada uma vira um contrato e guarda o que se sabia no momento do pedido
cust = rng.integers(1, N_CUSTOMERS + 1, N_CONTRACTS)
granted = (START + pd.to_timedelta(rng.integers(60, DAYS - 30, N_CONTRACTS), unit="D")).normalize()
amount = np.round(income0[cust - 1] * rng.uniform(1.5, 10, N_CONTRACTS), 2)
term = rng.choice([6, 12, 18, 24], N_CONTRACTS)
limit_usage = np.round(np.clip(rng.beta(2, 3, N_CONTRACTS) + 0.08 * hidden_risk[cust - 1], 0, 1), 3)
bureau_queries = rng.poisson(1.0 + hidden_risk[cust - 1])
recent_requests = 1 + rng.poisson(0.4 * hidden_risk[cust - 1])
proposals = pd.DataFrame({"contrato_id": np.arange(1, N_CONTRACTS + 1), "cliente_id": cust,
                          "pedido_em": granted - pd.to_timedelta(rng.integers(30, 600, N_CONTRACTS), unit="m"),
                          "valor_solicitado": amount, "prazo_meses": term, "uso_limite": limit_usage,
                          "consultas_bureau_90d": bureau_queries, "pedidos_ultimas_24h": recent_requests})

# quem vai deixar de pagar: o alvo que o modelo tenta prever
burden = np.clip((amount / term) / income0[cust - 1], 0, 3)
logit = (-5.6 + 2.2 * burden + 1.2 * limit_usage + 0.25 * bureau_queries + 0.4 * recent_requests
         + 0.6 * hidden_risk[cust - 1] - 0.6 * (segment[cust - 1] == "consignado"))
defaults = rng.random(N_CONTRACTS) < 1 / (1 + np.exp(-logit))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Eventos de CDC dos contratos

# COMMAND ----------

product = np.array([{"varejo": "pessoal", "alta_renda": "cartao", "consignado": "consignado"}[s] for s in segment[cust - 1]])
contracts = pd.DataFrame({"contrato_id": proposals["contrato_id"], "cliente_id": cust, "produto": product,
                          "valor": amount, "prazo_meses": term, "data_concessao": granted})
late = contracts[defaults]
renegotiated = late.sample(frac=0.3, random_state=1)
cancelled = contracts[~defaults].sample(n=15, random_state=1)
cdc = pd.concat([
    contracts.assign(status="em_dia", operacao="INSERT", evento_em=contracts["data_concessao"] + pd.Timedelta(hours=10)),
    late.assign(status="em_atraso", operacao="UPDATE",
                evento_em=late["data_concessao"] + pd.to_timedelta(rng.integers(60, 200, len(late)), unit="D")),
    renegotiated.assign(status="renegociado", operacao="UPDATE",
                        evento_em=renegotiated["data_concessao"] + pd.to_timedelta(rng.integers(220, 300, len(renegotiated)), unit="D")),
    cancelled.assign(status="cancelado", operacao="DELETE", evento_em=cancelled["data_concessao"] + pd.Timedelta(days=2)),
], ignore_index=True)
cdc["recebido_em"] = cdc["evento_em"] + pd.to_timedelta(rng.integers(1, 120, len(cdc)), unit="m")
out_of_order = cdc[cdc["status"] == "em_atraso"].sample(frac=0.05, random_state=1).index   # chegam 150 dias depois
cdc.loc[out_of_order, "recebido_em"] += pd.Timedelta(days=150)
cdc = cdc[cdc["evento_em"] <= END]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parcelas, exportadas "sujas" pelo core bancário

# COMMAND ----------

rows = []
for contract_id, is_default, months, grant_date, value in zip(contracts["contrato_id"], defaults, term, granted, amount):
    stops_paying = rng.integers(1, 7) if is_default else 999
    for k in range(1, months + 1):
        due = grant_date + pd.DateOffset(months=k)
        if due > END:
            break
        paid = pd.NaT if k >= stops_paying else due + pd.Timedelta(days=int(max(0, rng.normal(2, 6) + 15 * is_default * rng.random())))
        rows.append((contract_id, k, due, round(value / months, 2), paid if paid is not pd.NaT and paid <= END else pd.NaT))
installments = pd.DataFrame(rows, columns=["contrato_id", "numero", "vencimento", "valor", "pago_em"])
installments.insert(0, "parcela_id", np.arange(1, len(installments) + 1))
installments["cpf"] = customers.set_index("cliente_id").loc[
    contracts.set_index("contrato_id").loc[installments["contrato_id"], "cliente_id"], "cpf"].values

brazilian = rng.random(len(installments)) < 0.5                       # metade no formato brasileiro
installments["valor"] = [("R$ " + f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")) if b else f"{v:.2f}"
                         for v, b in zip(installments["valor"], brazilian)]
installments["vencimento"] = np.where(brazilian, installments["vencimento"].dt.strftime("%d/%m/%Y"),
                                      installments["vencimento"].dt.strftime("%Y-%m-%d"))
installments["pago_em"] = installments["pago_em"].dt.strftime("%Y-%m-%d")
orphans = installments.sample(n=20, random_state=2).assign(contrato_id=None)
orphans["parcela_id"] = np.arange(len(installments) + 1, len(installments) + 21)
installments = pd.concat([installments, installments.sample(frac=0.01, random_state=3), orphans], ignore_index=True)
installments = installments.astype({"contrato_id": "Int64"}).astype(str).replace({"<NA>": "", "nan": "", "None": ""})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Propostas novas, ainda sem contrato
# MAGIC Pedidos de 2026 de clientes que já existem, numa piora econômica (mais uso do limite e mais
# MAGIC consultas ao bureau que no histórico): o drift que o monitoramento deve pegar. Cerca de 1,5%
# MAGIC chegam sujas, para o contrato de entrada barrar no scoring e no endpoint.

# COMMAND ----------

N_PENDING = 5000
pending_cust = rng.integers(1, N_CUSTOMERS + 1, N_PENDING)
pending = pd.DataFrame({
    "contrato_id": np.arange(N_CONTRACTS + 1, N_CONTRACTS + N_PENDING + 1),   # número reservado; sem contrato ainda
    "cliente_id": pending_cust,
    "pedido_em": pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 90 * 24 * 60, N_PENDING), unit="m"),
    "valor_solicitado": np.round(income0[pending_cust - 1] * rng.uniform(1.5, 10, N_PENDING), 2),
    "prazo_meses": rng.choice([6, 12, 18, 24], N_PENDING),
    "uso_limite": np.round(np.clip(rng.beta(2, 3, N_PENDING) + 0.08 * hidden_risk[pending_cust - 1] + 0.25, 0, 1), 3),
    "consultas_bureau_90d": rng.poisson(2 * (1.0 + hidden_risk[pending_cust - 1])),
    "pedidos_ultimas_24h": 1 + rng.poisson(0.4 * hidden_risk[pending_cust - 1]),
})
for column, dirty_value in {"prazo_meses": 0, "valor_solicitado": -1500.0, "uso_limite": 87.0}.items():
    pending.loc[rng.choice(pending.index, 25, replace=False), column] = dirty_value

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gravar na landing (uma vez só)

# COMMAND ----------

BATCH = "lote_0001"
if any(f.name.startswith("parcelas") for f in dbutils.fs.ls(LANDING)):
    print("A landing já tem dados: nada foi gravado.")
else:
    spark.createDataFrame(installments).write.option("header", True).csv(f"{LANDING}/parcelas/{BATCH}")
    for name, df in {"contratos_cdc": cdc, "clientes": customers, "renda": income, "propostas": proposals}.items():
        spark.createDataFrame(df).write.json(f"{LANDING}/{name}/{BATCH}")
    spark.createDataFrame(pending).write.json(f"{LANDING}/propostas/lote_0002")
    print("Dados gravados. O job credit_ingestion dispara sozinho com a chegada dos arquivos.")
display(dbutils.fs.ls(LANDING))
