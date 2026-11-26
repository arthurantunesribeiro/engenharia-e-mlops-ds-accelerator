"""Monitoramento: classificação do drift calculado pelo data profiling e alertas."""
from __future__ import annotations

import pandas as pd

OUT_OF_CONTRACT_LIMIT = 0.02  # acima de 2% de chamadas recusadas ou propostas em quarentena, alguém olha


def classify_psi(value: float) -> str:
    """Classifica o PSI com a régua do data profiling do Databricks.

    Args:
        value: PSI de uma coluna (`population_stability_index` da tabela de drift).

    Returns:
        `"estável"` (< 0,1), `"atenção"` (< 0,2) ou `"investigar"`.
    """
    if value < 0.1:
        return "estável"
    return "atenção" if value < 0.2 else "investigar"


def build_alerts(drift: pd.DataFrame, out_of_contract_rate: float = 0.0) -> list[str]:
    """Transforma o drift e a qualidade da entrada em mensagens de alerta, com o mais grave primeiro.

    Dado fora do contrato vem antes do drift: se a entrada quebrou, o PSI mede o defeito, não o mundo.

    Args:
        drift: Uma linha por coluna e segmento, com `column_name`, `slice_value` (nulo = base inteira) e `psi`.
        out_of_contract_rate: Fração do tráfego recusada ou em quarentena.

    Returns:
        Lista de mensagens (vazia se tudo estável).
    """
    messages = []
    if out_of_contract_rate > OUT_OF_CONTRACT_LIMIT:
        messages.append(f"{out_of_contract_rate:.1%} do tráfego fora do contrato de entrada "
                        f"(limite {OUT_OF_CONTRACT_LIMIT:.0%}): investigar a origem antes do drift")
    unstable = drift[drift["psi"].map(classify_psi) != "estável"].sort_values("psi", ascending=False)
    for _, r in unstable.iterrows():
        scope = "base inteira" if pd.isna(r["slice_value"]) else f"segmento {r['slice_value']}"
        messages.append(f"PSI de {r['column_name']} ({scope}) = {r['psi']:.3f} ({classify_psi(r['psi'])})")
    return messages
