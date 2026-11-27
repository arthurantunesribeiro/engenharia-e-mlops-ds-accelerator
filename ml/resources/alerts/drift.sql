WITH baseline_drift AS (
  SELECT window.start AS inicio, column_name, coalesce(slice_value, 'base inteira') AS fatia,
         population_stability_index AS psi
  FROM main.credit_ml.credit_score_inference_log_drift_metrics
  WHERE drift_type = 'BASELINE' AND population_stability_index IS NOT NULL
)
SELECT round(max(psi), 3) AS maior_psi,
       max_by(concat(column_name, ' (', fatia, ')'), psi) AS onde,
       count_if(psi > 0.2) AS colunas_acima_de_0_2
FROM baseline_drift
WHERE inicio = (SELECT max(inicio) FROM baseline_drift)
