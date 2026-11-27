-- BRONZE: como chegou, cada arquivo lido uma vez, com a origem e a hora da carga

CREATE OR REFRESH STREAMING TABLE bronze_parcelas
COMMENT 'Parcelas exportadas pelo core bancário, tudo como texto'
AS SELECT *, _metadata.file_path AS _arquivo, current_timestamp() AS _carga_em
FROM STREAM read_files('${landing}/parcelas', format => 'csv', header => true, inferSchema => false);

CREATE OR REFRESH STREAMING TABLE bronze_contratos_cdc
COMMENT 'Eventos de CDC dos contratos: INSERT, UPDATE e DELETE'
AS SELECT *, _metadata.file_path AS _arquivo, current_timestamp() AS _carga_em
FROM STREAM read_files('${landing}/contratos_cdc', format => 'json',
  schema => 'contrato_id BIGINT, cliente_id BIGINT, produto STRING, valor DOUBLE, prazo_meses INT, data_concessao TIMESTAMP, status STRING, operacao STRING, evento_em TIMESTAMP, recebido_em TIMESTAMP');

CREATE OR REFRESH STREAMING TABLE bronze_clientes
AS SELECT * FROM STREAM read_files('${landing}/clientes', format => 'json',
  schema => 'cliente_id BIGINT, cpf STRING, segmento STRING, canal STRING, cadastrado_em TIMESTAMP');

CREATE OR REFRESH STREAMING TABLE bronze_renda
AS SELECT * FROM STREAM read_files('${landing}/renda', format => 'json',
  schema => 'cliente_id BIGINT, renda_mensal DOUBLE, atualizado_em TIMESTAMP');

CREATE OR REFRESH STREAMING TABLE bronze_propostas
AS SELECT * FROM STREAM read_files('${landing}/propostas', format => 'json',
  schema => 'contrato_id BIGINT, cliente_id BIGINT, pedido_em TIMESTAMP, valor_solicitado DOUBLE, prazo_meses INT, uso_limite DOUBLE, consultas_bureau_90d INT, pedidos_ultimas_24h INT');


-- SILVER: limpo, tipado e confiável

-- Parcelas tipadas: valor brasileiro vira decimal, duas datas viram uma, CPF vira hash
CREATE TEMPORARY VIEW parcelas_tipadas AS
SELECT
  try_cast(parcela_id AS BIGINT)                                         AS parcela_id,
  try_cast(contrato_id AS BIGINT)                                        AS contrato_id,
  try_cast(numero AS INT)                                                AS numero,
  CASE WHEN valor LIKE '%,%'
       THEN try_cast(replace(replace(replace(valor, 'R$ ', ''), '.', ''), ',', '.') AS DECIMAL(12,2))
       ELSE try_cast(valor AS DECIMAL(12,2)) END                         AS valor,
  to_date(coalesce(try_to_timestamp(vencimento, 'dd/MM/yyyy'),
                   try_to_timestamp(vencimento, 'yyyy-MM-dd')))          AS vencimento,
  to_date(try_to_timestamp(pago_em, 'yyyy-MM-dd'))                       AS pago_em,
  sha2(cpf, 256)                                                         AS cpf_hash,
  _carga_em
FROM bronze_parcelas;

-- Linha inválida descartada pelas regras; duplicada sai (fica a carga mais recente)
CREATE OR REFRESH MATERIALIZED VIEW silver_parcelas (
  CONSTRAINT contrato_presente      EXPECT (contrato_id IS NOT NULL) ON VIOLATION DROP ROW,
  CONSTRAINT valor_positivo         EXPECT (valor > 0)               ON VIOLATION DROP ROW,
  CONSTRAINT vencimento_reconhecido EXPECT (vencimento IS NOT NULL)  ON VIOLATION DROP ROW
)
COMMENT 'Uma linha por parcela, tipada e sem CPF'
AS SELECT parcela_id, contrato_id, numero, valor, vencimento, pago_em, cpf_hash
FROM (SELECT *, row_number() OVER (PARTITION BY parcela_id ORDER BY _carga_em DESC) AS versao FROM parcelas_tipadas)
WHERE versao = 1;

-- Quarentena: as mesmas regras ao contrário, com o motivo
CREATE OR REFRESH MATERIALIZED VIEW quarentena_parcelas
COMMENT 'Parcelas que violaram alguma regra, com o motivo'
AS SELECT *,
  concat_ws(', ',
    CASE WHEN contrato_id IS NULL THEN 'contrato_presente' END,
    CASE WHEN coalesce(valor, 0) <= 0 THEN 'valor_positivo' END,
    CASE WHEN vencimento IS NULL THEN 'vencimento_reconhecido' END) AS regras_violadas
FROM parcelas_tipadas
WHERE contrato_id IS NULL OR coalesce(valor, 0) <= 0 OR vencimento IS NULL;

-- Contratos: estado atual pela ordem do fato (evento_em), não da chegada (SCD 1)
CREATE OR REFRESH STREAMING TABLE silver_contratos
COMMENT 'Estado atual de cada contrato';

CREATE FLOW cdc_contratos AS AUTO CDC INTO silver_contratos
FROM STREAM(bronze_contratos_cdc)
KEYS (contrato_id)
APPLY AS DELETE WHEN operacao = 'DELETE'
SEQUENCE BY evento_em
COLUMNS * EXCEPT (operacao, recebido_em, _arquivo, _carga_em)
STORED AS SCD TYPE 1;

-- Renda: cada versão com o período em que valeu (SCD 2)
CREATE OR REFRESH STREAMING TABLE silver_renda
COMMENT 'Histórico de renda de cada cliente';

CREATE FLOW cdc_renda AS AUTO CDC INTO silver_renda
FROM STREAM(bronze_renda)
KEYS (cliente_id)
SEQUENCE BY atualizado_em
STORED AS SCD TYPE 2;


-- GOLD: esquema estrela, o indicador do negócio e as tabelas do modelo

-- Dimensão: uma linha por cliente, com a renda atual
CREATE OR REFRESH MATERIALIZED VIEW dim_cliente
COMMENT 'Uma linha por cliente'
AS SELECT c.cliente_id, sha2(c.cpf, 256) AS cpf_hash, c.segmento, c.canal, r.renda_mensal
FROM bronze_clientes c
LEFT JOIN silver_renda r ON r.cliente_id = c.cliente_id AND r.__END_AT IS NULL;

-- Fato: chaves e medidas; atraso medido na data de corte (31/12/2025)
CREATE OR REFRESH MATERIALIZED VIEW fato_parcelas
COMMENT 'Uma linha por parcela'
AS SELECT p.parcela_id, p.contrato_id, k.cliente_id, p.vencimento AS data_vencimento, p.numero, p.valor, p.pago_em,
  greatest(0, datediff(coalesce(p.pago_em, DATE'2025-12-31'), p.vencimento)) AS dias_atraso,
  CASE WHEN datediff(coalesce(p.pago_em, DATE'2025-12-31'), p.vencimento) >= 90 THEN 1 ELSE 0 END AS atraso_90
FROM silver_parcelas p
JOIN silver_contratos k ON k.contrato_id = p.contrato_id;

-- Inadimplência 90d por safra e produto
CREATE OR REFRESH MATERIALIZED VIEW gold_inadimplencia_safra
COMMENT 'Inadimplência de 90 dias por safra e produto'
AS SELECT date_format(k.data_concessao, 'yyyy-MM') AS safra, k.produto,
  count(*) AS contratos, round(avg(m.inadimplente), 4) AS inadimplencia_90d
FROM (SELECT contrato_id, max(atraso_90) AS inadimplente FROM fato_parcelas WHERE numero <= 6 GROUP BY contrato_id) m
JOIN silver_contratos k USING (contrato_id)
GROUP BY 1, 2;

-- ABT point-in-time: uma linha por contrato maturado, com o que se sabia na concessão e o alvo
CREATE OR REFRESH MATERIALIZED VIEW gold_abt_credito (
  CONSTRAINT contrato_presente EXPECT (contrato_id IS NOT NULL)     ON VIOLATION FAIL UPDATE,
  CONSTRAINT renda_positiva    EXPECT (renda_mensal > 0)            ON VIOLATION FAIL UPDATE,
  CONSTRAINT uso_limite_valido EXPECT (uso_limite BETWEEN 0 AND 1)  ON VIOLATION FAIL UPDATE
)
COMMENT 'ABT point-in-time: features na concessão e o alvo inadimplente_90d'
AS WITH base AS (
  SELECT contrato_id, cliente_id, to_date(data_concessao) AS data_ref, valor AS valor_solicitado, prazo_meses
  FROM silver_contratos
  WHERE date_add(to_date(data_concessao), 276) <= DATE'2025-12-31'
),
renda_na_data AS (
  SELECT b.contrato_id, r.renda_mensal
  FROM base b JOIN silver_renda r
    ON r.cliente_id = b.cliente_id AND r.__START_AT <= b.data_ref AND (r.__END_AT IS NULL OR b.data_ref < r.__END_AT)
),
atraso_ate_a_data AS (
  SELECT b.contrato_id,
         round(avg(greatest(0, datediff(least(coalesce(f.pago_em, b.data_ref), b.data_ref), f.data_vencimento))), 1) AS atraso_medio_12m
  FROM base b JOIN fato_parcelas f
    ON f.cliente_id = b.cliente_id AND f.data_vencimento < b.data_ref AND f.data_vencimento >= date_sub(b.data_ref, 365)
  GROUP BY b.contrato_id
),
alvo AS (
  SELECT contrato_id, max(atraso_90) AS inadimplente_90d FROM fato_parcelas WHERE numero <= 6 GROUP BY contrato_id
)
SELECT b.contrato_id, b.cliente_id, b.data_ref, c.segmento, c.canal,
       r.renda_mensal, b.valor_solicitado, b.prazo_meses,
       p.uso_limite, coalesce(a.atraso_medio_12m, 0) AS atraso_medio_12m,
       p.consultas_bureau_90d, p.pedidos_ultimas_24h,
       coalesce(t.inadimplente_90d, 0) AS inadimplente_90d
FROM base b
JOIN dim_cliente c ON c.cliente_id = b.cliente_id
JOIN bronze_propostas p ON p.contrato_id = b.contrato_id
LEFT JOIN renda_na_data r ON r.contrato_id = b.contrato_id
LEFT JOIN atraso_ate_a_data a ON a.contrato_id = b.contrato_id
LEFT JOIN alvo t ON t.contrato_id = b.contrato_id;

-- Propostas sem contrato: as features da ABT na data do pedido; o contrato do modelo barra as sujas
CREATE OR REFRESH MATERIALIZED VIEW gold_propostas_pendentes
COMMENT 'Propostas sem contrato, com as features na data do pedido'
AS WITH pendentes AS (
  SELECT contrato_id AS proposta_id, cliente_id, to_date(pedido_em) AS data_ref, valor_solicitado, prazo_meses,
         uso_limite, consultas_bureau_90d, pedidos_ultimas_24h
  FROM bronze_propostas
  WHERE contrato_id NOT IN (SELECT contrato_id FROM bronze_contratos_cdc)
),
atraso AS (
  SELECT p.proposta_id,
         round(avg(greatest(0, datediff(least(coalesce(f.pago_em, p.data_ref), p.data_ref), f.data_vencimento))), 1) AS atraso_medio_12m
  FROM pendentes p JOIN fato_parcelas f
    ON f.cliente_id = p.cliente_id AND f.data_vencimento < p.data_ref AND f.data_vencimento >= date_sub(p.data_ref, 365)
  GROUP BY p.proposta_id
)
SELECT p.proposta_id, p.cliente_id, p.data_ref, c.segmento, c.renda_mensal, p.valor_solicitado, p.prazo_meses,
       p.uso_limite, coalesce(a.atraso_medio_12m, 0) AS atraso_medio_12m, p.consultas_bureau_90d, p.pedidos_ultimas_24h
FROM pendentes p
JOIN dim_cliente c ON c.cliente_id = p.cliente_id
LEFT JOIN atraso a ON a.proposta_id = p.proposta_id;
