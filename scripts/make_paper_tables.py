"""Generate the manuscript's data tables from the cached experiment artifacts.

Every number in these tables comes from a file under ``scripts/figures/``;
the manuscript ``\\input``s the generated ``.tex`` fragments, so a re-run of an
experiment followed by this script updates the paper with no hand editing.

Writes to ``<repo>/../Latex (Metascience)/tables/`` (override with --out).

Run:  python -m scripts.make_paper_tables
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent / "figures"
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "Latex (Metascience)" / "tables"


def _fmt_sci(x: float) -> str:
    if x == float("inf"):
        return r"$\infty$"
    m, e = f"{x:.2e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def embedding_table(out: Path) -> None:
    rows = list(csv.DictReader((FIG_DIR / "embedding_study.csv").open(encoding="utf-8")))
    by = {}
    for r in rows:
        by[(int(r["N"]), r["topology"], r["method"])] = r
    Ns = sorted({int(r["N"]) for r in rows})
    lines = [r"\begin{tabular}{r r cc cc cc}", r"\toprule",
             r"$N$ & Coef.\ range & \multicolumn{2}{c}{Pegasus P16, clique} & "
             r"\multicolumn{2}{c}{Zephyr Z12, clique} & \multicolumn{2}{c}{Pegasus, heuristic} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
             r" & & max chain & qubits & max chain & qubits & max chain & qubits \\", r"\midrule"]
    for N in Ns:
        def cell(topo, method):
            r = by.get((N, topo, method))
            if r is None or r["embedded"] != "True":
                return "--- & ---"
            return f"{int(float(r['max_chain']))} & {int(float(r['physical_qubits']))}"
        cr = float(by[(N, "Pegasus P16 (Advantage)", "clique")]["coefficient_range"])
        lines.append(f"{N} & {cr:,.0f} & {cell('Pegasus P16 (Advantage)', 'clique')} & "
                     f"{cell('Zephyr Z12 (Advantage2)', 'clique')} & "
                     f"{cell('Pegasus P16 (Advantage)', 'heuristic')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_embedding.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def mesh_table(out: Path) -> None:
    src = FIG_DIR / "collocation_mesh_study.json"
    d = json.loads(src.read_text(encoding="utf-8"))
    lines = [r"\begin{tabular}{r c c c c}", r"\toprule",
             r"$N$ & $J$ & max defect & feasible & SLSQP iterations \\", r"\midrule"]
    for r in d["rows"]:
        lines.append(f"{r['N']} & {r['J']:.4f} & {_fmt_sci(r['max_defect'])} & "
                     f"{'yes' if r['feasible'] else 'no'} & {r['iterations']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_mesh.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if "richardson" in d:
        rr = d["richardson"]
        (out / "mesh_richardson.tex").write_text(
            f"{rr['J_extrapolated']:.3f}", encoding="utf-8")


def tts_table(out: Path) -> None:
    src = FIG_DIR / "scaling_results.json"
    rows = json.loads(src.read_text(encoding="utf-8"))
    lines = [r"\begin{tabular}{r l r r r r r}", r"\toprule",
             r"$N$ & Solver & Best energy & $p$ & $t_{\mathrm{run}}$ (s) & TTT$_{99}$ (s) & Reference \\",
             r"\midrule"]
    label = {"sa": "SA (100 reads)", "sb": "SB (100 agents)", "kerberos": "Kerberos (3 it.)"}
    for r in rows:
        N = r["N"]
        ref = r.get("reference", {})
        refk = ref.get("kind", "")
        res = r["results"]
        if "brute_force" in res and "energy" in res["brute_force"]:
            bf = res["brute_force"]
            lines.append(f"{N} & Brute force & {_fmt_sci(bf['energy'])} & 1 & {bf['time']:.2f} & {bf['time']:.2f} & exact \\\\")
        if "milp" in res and "energy" in res["milp"]:
            m = res["milp"]
            note = "optimal" if m.get("is_optimal") else "time limit"
            lines.append(f"{N} & MILP/HiGHS & {_fmt_sci(m['energy'])} & --- & {m['time']:.1f} & --- & {note} \\\\")
        for k in ("sa", "sb", "kerberos"):
            t = r.get("tts", {}).get(k)
            if not t:
                continue
            lines.append(f"{N} & {label[k]} & {_fmt_sci(t['best'])} & {t['p']:.2f} & "
                         f"{t['t_run']:.2f} & {_fmt_sci(t['tts99']) if t['tts99'] == float('inf') else f'{t['tts99']:.2f}'} & {refk} \\\\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_tts.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    embedding_table(args.out)
    mesh_table(args.out)
    tts_table(args.out)
    print("tables written to", args.out)


if __name__ == "__main__":
    main()
