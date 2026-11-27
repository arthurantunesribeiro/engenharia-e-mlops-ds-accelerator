from credit_score.features import INPUT_COLUMNS
from credit_score.monitoring import PROFILE_COLUMN_TYPES, chat_message, profile_columns


def test_profile_columns_cast_to_the_baseline_types():
    exprs = profile_columns(["prazo_meses", "score", "segmento"])
    assert exprs == ["CAST(prazo_meses AS int) AS prazo_meses", "CAST(score AS double) AS score",
                     "CAST(segmento AS string) AS segmento"]


def test_profile_types_cover_every_model_input():
    assert set(INPUT_COLUMNS) | {"score", "faixa", "versao_modelo"} <= set(PROFILE_COLUMN_TYPES)


DRIFT_HIGH = {"maior_psi": 2.299, "onde": "uso_limite (alta_renda)", "colunas_acima_de_0_2": 17}
CONTRACT_OK = {"maior_taxa_fora_do_contrato": "0.0196", "taxa_endpoint_7d": "0.0196", "taxa_ultimo_lote": "0.015"}


def test_chat_message_reports_drift_above_the_limit():
    message = chat_message(DRIFT_HIGH, CONTRACT_OK)
    assert "PSI 2.30* em uso_limite (alta_renda)" in message and "17 coluna" in message
    assert "fora do contrato" not in message


def test_out_of_contract_comes_before_drift():
    message = chat_message(DRIFT_HIGH, {**CONTRACT_OK, "maior_taxa_fora_do_contrato": 0.08, "taxa_endpoint_7d": 0.08})
    assert message.index("fora do contrato") < message.index("Drift")


def test_no_message_when_everything_is_within_limits():
    assert chat_message({"maior_psi": 0.05, "onde": "score (base inteira)"}, CONTRACT_OK) is None
    assert chat_message({"maior_psi": None}, {"maior_taxa_fora_do_contrato": None}) is None
