"""
Selección aleatoria y reproducible de una muestra de fondos de funds.db
para ampliar el golden dataset de extracción.

Uso:
  python -m evaluation.seleccionar_muestra_golden
  python -m evaluation.seleccionar_muestra_golden --n 100 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from pathlib import Path

DB_PATH = Path("database/funds.db")
OUTPUT_PATH = Path("evaluation/muestra_golden_100.json")


def seleccionar_muestra(db_path: Path, n: int, seed: int) -> list[dict]:
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        filas = con.execute(
            "SELECT isin, nombre_fondo, pdf_origen FROM funds ORDER BY isin"
        ).fetchall()

    random.seed(seed)
    muestra = random.sample(filas, n)
    return [dict(fila) for fila in muestra]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Selecciona una muestra aleatoria reproducible de fondos de funds.db."
    )
    parser.add_argument("--n", type=int, default=100, metavar="N")
    parser.add_argument("--seed", type=int, default=42, metavar="SEED")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    muestra = seleccionar_muestra(args.db, args.n, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(muestra, f, ensure_ascii=False, indent=2)

    print(f"Muestra de {len(muestra)} fondos (seed={args.seed}) guardada en: {args.out}")


if __name__ == "__main__":
    main()
