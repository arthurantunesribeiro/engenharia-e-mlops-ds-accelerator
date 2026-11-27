"""Catálogo do Unity Catalog onde ficam os schemas `credit` e `credit_ml`."""

from __future__ import annotations

import argparse

DEFAULT_CATALOG = "workspace"


def catalog_from_args(argv: list[str] | None = None) -> str:
    """Lê o catálogo do parâmetro `--catalog` que o bundle passa aos jobs.

    Args:
        argv: Argumentos da linha de comando (padrão: os do processo).

    Returns:
        O catálogo informado, ou `DEFAULT_CATALOG`.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    return parser.parse_known_args(argv)[0].catalog
