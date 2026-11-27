from credit_score.features import INPUT_COLUMNS
from credit_score.monitoring import PROFILE_COLUMN_TYPES, profile_columns


def test_profile_columns_cast_to_the_baseline_types():
    exprs = profile_columns(["prazo_meses", "score", "segmento"])
    assert exprs == [
        "CAST(prazo_meses AS int) AS prazo_meses",
        "CAST(score AS double) AS score",
        "CAST(segmento AS string) AS segmento",
    ]


def test_profile_types_cover_every_model_input():
    assert set(INPUT_COLUMNS) | {"score", "faixa", "versao_modelo"} <= set(PROFILE_COLUMN_TYPES)
