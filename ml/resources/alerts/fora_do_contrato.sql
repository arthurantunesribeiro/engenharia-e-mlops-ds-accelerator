WITH endpoint AS (
  SELECT avg(CASE WHEN status_code = 400 THEN 1.0 ELSE 0.0 END) AS taxa
  FROM main.credit_ml.chamadas_endpoint
  WHERE request_time >= current_timestamp() - INTERVAL 7 DAYS
),
ultimo_lote AS (
  SELECT count(*) AS barradas
  FROM main.credit_ml.propostas_quarentena
  WHERE barrado_em = (SELECT max(barrado_em) FROM main.credit_ml.propostas_quarentena)
),
batch AS (
  SELECT barradas / (barradas + (SELECT count(*) FROM main.credit_ml.scores_credito)) AS taxa FROM ultimo_lote
)
SELECT round(greatest(coalesce((SELECT taxa FROM endpoint), 0), coalesce((SELECT taxa FROM batch), 0)), 4)
         AS maior_taxa_fora_do_contrato,
       round((SELECT taxa FROM endpoint), 4) AS taxa_endpoint_7d,
       round((SELECT taxa FROM batch), 4) AS taxa_ultimo_lote
