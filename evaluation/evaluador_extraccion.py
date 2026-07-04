"""
Evaluador de calidad de extracción (Evaluador 1).

Compara los datos extraídos por Gemini y almacenados en SQLite
contra los valores reales definidos en el golden dataset.

Uso:
  python -m evaluation.evaluador_extraccion
  python -m evaluation.evaluador_extraccion --golden evaluation/golden_dataset_extraccion.json
  python -m evaluation.evaluador_extraccion --db database/funds.db --verbose
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

GOLDEN_PATH = Path("evaluation/golden_dataset_extraccion.json")
DB_PATH     = Path("database/funds.db")

CAMPOS_NUMERICOS = {
    "nivel_riesgo",
    "horizonte_recomendado_anios",
    "comision_gestion",
    "comision_suscripcion",
    "comision_reembolso",
    "importe_minimo_inversion",
}

CAMPOS_BOOLEANOS = {"esg"}
CAMPOS_TEXTO     = {"tipo_gestion"}
TOLERANCIA_NUMERICA = 0.05   # ±5% de tolerancia para campos numéricos


# ── Resultado por campo ───────────────────────────────────────────────────────

@dataclass
class ResultadoCampo:
    campo:     str
    esperado:  object
    obtenido:  object
    correcto:  bool
    motivo:    str = ""


@dataclass
class ResultadoFondo:
    isin:              str
    nombre_fondo:      str
    campos:            list[ResultadoCampo] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len([c for c in self.campos if c.esperado is not None])

    @property
    def correctos(self) -> int:
        return sum(1 for c in self.campos if c.correcto and c.esperado is not None)

    @property
    def precision(self) -> float:
        return self.correctos / self.total if self.total > 0 else 0.0


# ── Carga de datos ────────────────────────────────────────────────────────────

def cargar_golden(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cargar_fondo_bd(isin: str, db_path: Path) -> dict | None:
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM funds WHERE isin = ?", (isin,)).fetchone()
    return dict(row) if row else None


# ── Comparadores ──────────────────────────────────────────────────────────────

def _comparar_numerico(esperado: float, obtenido) -> tuple[bool, str]:
    if obtenido is None:
        return False, "campo null en BD"
    try:
        ob = float(obtenido)
        es = float(esperado)
        if es == 0:
            return ob == 0, f"esperado=0, obtenido={ob}"
        diferencia = abs(ob - es) / abs(es)
        ok = diferencia <= TOLERANCIA_NUMERICA
        return ok, f"esperado={es}, obtenido={ob}, diff={diferencia:.1%}"
    except (TypeError, ValueError):
        return False, f"no se pudo convertir: {obtenido}"


def _comparar_booleano(esperado: bool, obtenido) -> tuple[bool, str]:
    if obtenido is None:
        return False, "campo null en BD"
    ob_bool = bool(obtenido)
    ok = ob_bool == esperado
    return ok, f"esperado={esperado}, obtenido={ob_bool}"


def _comparar_texto(esperado: str, obtenido) -> tuple[bool, str]:
    if obtenido is None:
        return False, "campo null en BD"
    ok = esperado.lower().strip() in obtenido.lower().strip()
    return ok, f"esperado='{esperado}', obtenido='{obtenido}'"


def _comparar_politica(palabras_clave: list[str], obtenido: str | None) -> tuple[bool, str]:
    if not palabras_clave:
        return True, "sin palabras clave definidas"
    if not obtenido:
        return False, "politica_inversion null en BD"
    texto = obtenido.lower()
    faltantes = [p for p in palabras_clave if p.lower() not in texto]
    ok = len(faltantes) == 0
    motivo = "OK" if ok else f"palabras no encontradas: {faltantes}"
    return ok, motivo


# ── Evaluación de un fondo ────────────────────────────────────────────────────

def evaluar_fondo(entrada_golden: dict, db_path: Path) -> ResultadoFondo:
    isin   = entrada_golden["isin"]
    nombre = entrada_golden["nombre_fondo"]
    result = ResultadoFondo(isin=isin, nombre_fondo=nombre)

    fondo_bd = cargar_fondo_bd(isin, db_path)
    if fondo_bd is None:
        result.campos.append(ResultadoCampo(
            campo="*", esperado="fondo en BD", obtenido=None,
            correcto=False, motivo="ISIN no encontrado en la base de datos"
        ))
        return result

    esperado = entrada_golden["esperado"]

    for campo, valor_esperado in esperado.items():
        if valor_esperado is None:
            continue   # campo no definido en el golden → no evaluar

        if campo == "politica_inversion_contiene":
            ok, motivo = _comparar_politica(valor_esperado, fondo_bd.get("politica_inversion"))
            result.campos.append(ResultadoCampo(
                campo=campo, esperado=valor_esperado,
                obtenido=fondo_bd.get("politica_inversion", "")[:80] + "…",
                correcto=ok, motivo=motivo
            ))

        elif campo in CAMPOS_NUMERICOS:
            ok, motivo = _comparar_numerico(valor_esperado, fondo_bd.get(campo))
            result.campos.append(ResultadoCampo(
                campo=campo, esperado=valor_esperado,
                obtenido=fondo_bd.get(campo), correcto=ok, motivo=motivo
            ))

        elif campo in CAMPOS_BOOLEANOS:
            ok, motivo = _comparar_booleano(valor_esperado, fondo_bd.get(campo))
            result.campos.append(ResultadoCampo(
                campo=campo, esperado=valor_esperado,
                obtenido=fondo_bd.get(campo), correcto=ok, motivo=motivo
            ))

        elif campo in CAMPOS_TEXTO:
            ok, motivo = _comparar_texto(str(valor_esperado), str(fondo_bd.get(campo, "")))
            result.campos.append(ResultadoCampo(
                campo=campo, esperado=valor_esperado,
                obtenido=fondo_bd.get(campo), correcto=ok, motivo=motivo
            ))

    return result


# ── Informe ───────────────────────────────────────────────────────────────────

def imprimir_informe(resultados: list[ResultadoFondo], verbose: bool = False) -> None:
    sep = "─" * 70

    print("\n" + "═" * 70)
    print("  EVALUADOR 1 — CALIDAD DE EXTRACCIÓN")
    print("═" * 70)

    total_campos   = 0
    campos_ok      = 0
    conteo_campos: dict[str, dict] = {}

    for r in resultados:
        print(f"\n{r.nombre_fondo} ({r.isin})")
        print(f"  Precisión: {r.correctos}/{r.total}  ({r.precision:.0%})")

        for c in r.campos:
            if c.esperado is None:
                continue
            estado = "✓" if c.correcto else "✗"
            if verbose or not c.correcto:
                print(f"  {estado} {c.campo:<35} {c.motivo}")

            # Acumular por campo
            if c.campo not in conteo_campos:
                conteo_campos[c.campo] = {"ok": 0, "total": 0}
            conteo_campos[c.campo]["total"] += 1
            if c.correcto:
                conteo_campos[c.campo]["ok"] += 1

            total_campos += 1
            if c.correcto:
                campos_ok += 1

        print(sep)

    # Resumen global
    precision_global = campos_ok / total_campos if total_campos > 0 else 0.0
    print(f"\nRESUMEN GLOBAL")
    print(f"  Fondos evaluados : {len(resultados)}")
    print(f"  Campos evaluados : {total_campos}")
    print(f"  Precisión global : {campos_ok}/{total_campos}  ({precision_global:.0%})")

    # Precisión por campo
    print(f"\nPRECISIÓN POR CAMPO:")
    for campo, counts in sorted(conteo_campos.items(), key=lambda x: x[1]["ok"] / x[1]["total"]):
        p = counts["ok"] / counts["total"]
        barra = "█" * int(p * 20)
        print(f"  {campo:<35} {counts['ok']}/{counts['total']}  {barra} {p:.0%}")


def guardar_json(resultados: list[ResultadoFondo], output_path: Path) -> None:
    data = []
    for r in resultados:
        data.append({
            "isin": r.isin,
            "nombre_fondo": r.nombre_fondo,
            "precision": round(r.precision, 4),
            "correctos": r.correctos,
            "total": r.total,
            "campos": [
                {
                    "campo": c.campo,
                    "esperado": c.esperado,
                    "obtenido": c.obtenido,
                    "correcto": c.correcto,
                    "motivo": c.motivo,
                }
                for c in r.campos if c.esperado is not None
            ],
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nResultados guardados en: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluador de calidad de extracción de datos de fondos."
    )
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--db",     type=Path, default=DB_PATH)
    parser.add_argument("--output", type=Path, default=Path("evaluation/resultados_extraccion.json"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.golden.exists():
        print(f"Golden dataset no encontrado: {args.golden}")
        return
    if not args.db.exists():
        print(f"Base de datos no encontrada: {args.db}")
        return

    golden    = cargar_golden(args.golden)
    resultados = [evaluar_fondo(entrada, args.db) for entrada in golden]

    imprimir_informe(resultados, verbose=args.verbose)
    guardar_json(resultados, args.output)


if __name__ == "__main__":
    main()
