import sqlite3

import pandas as pd
import pytest
from conftest import PROPOSAL, proposal

from credit_score.contract import contract_sql_rules, validate_inputs

OUT_OF_CONTRACT = [
    ({"uso_limite": 7.0}, "uso_limite_valid"),
    ({"segmento": "premium"}, "segmento_valid"),
    ({"prazo_meses": 0}, "prazo_meses_valid"),
    ({"renda_mensal": None}, "renda_mensal_valid"),
    ({"valor_solicitado": -500.0}, "valor_solicitado_valid"),
    ({"consultas_bureau_90d": -1}, "consultas_bureau_90d_valid"),
]


def test_valid_proposal_passes():
    assert validate_inputs(proposal()).iloc[0] == []


@pytest.mark.parametrize("changes,rule", OUT_OF_CONTRACT)
def test_broken_contract_names_the_rule(changes, rule):
    assert validate_inputs(proposal(**changes)).iloc[0] == [rule]


def test_missing_column_fails_its_rule():
    assert validate_inputs(proposal().drop(columns=["renda_mensal"])).iloc[0] == ["renda_mensal_valid"]


def test_sql_rules_agree_with_pandas():
    """As duas formas do contrato (endpoint e batch) dizem o mesmo sobre as mesmas linhas."""
    rows = pd.DataFrame([PROPOSAL] + [{**PROPOSAL, **c} for c, _ in OUT_OF_CONTRACT])
    con = sqlite3.connect(":memory:")
    rows.to_sql("proposals", con, index=False)
    for name, expr in contract_sql_rules().items():
        sql = [bool(v) for (v,) in con.execute(f"SELECT coalesce(({expr}), 0) FROM proposals")]
        pandas = [name not in f for f in validate_inputs(rows)]
        assert sql == pandas, name
