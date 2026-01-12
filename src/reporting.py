from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

from .config import POLICIES
from .config import POLICY_GROUP_1, POLICY_GROUP_2

def _fmt_mean_std(vals: List[float]) -> str:
    if not vals:
        return "-"
    return f"{float(np.mean(vals)):.4f} ± {float(np.std(vals)):.4f}"


def save_hv_tables(out_dir: Path, scenario_name: str, hv_stats: Dict[int, Dict[str, List[float]]]) -> None:
    """Genera tablas como en P2 (2 tablas):

    - Tabla 1: FIFO, LTP, STP
    - Tabla 2: RRFIFO, RRLTP, RRECA

    Cada celda se reporta como "mean ± std" (HV) para gen 20/40/60/80/100.
    También se guarda CSV con mean/std por separado.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    gens = sorted(hv_stats.keys())

    def _write_group(group: List[str], tag: str) -> None:
        # Markdown
        md_path = out_dir / f"hv_table_{tag}_{scenario_name}.md"
        lines = []
        header = "| Política | " + " | ".join([f"Gen {g}" for g in gens]) + " |"
        sep = "|" + "---|" * (len(gens) + 1)
        lines.extend([header, sep])
        for pol in group:
            row = [pol] + [_fmt_mean_std(hv_stats[g][pol]) for g in gens]
            lines.append("| " + " | ".join(row) + " |")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # CSV (mean/std separados)
        csv_path = out_dir / f"hv_table_{tag}_{scenario_name}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["policy", "generation", "mean", "std", "min", "max"])
            for pol in group:
                for g in gens:
                    vals = hv_stats[g][pol]
                    if not vals:
                        continue
                    w.writerow([pol, g, float(np.mean(vals)), float(np.std(vals)), float(np.min(vals)), float(np.max(vals))])

    _write_group(POLICY_GROUP_1, tag="grp1")
    _write_group(POLICY_GROUP_2, tag="grp2")


def select_extremes_and_knee(points: List[Tuple[float, float]]) -> Tuple[int, int, int]:
    """Selecciona 3 puntos de un frente (minimización):
    - Extremo makespan: mínimo en obj1
    - Extremo energía: mínimo en obj2
    - Rodilla: máxima distancia a la recta entre extremos (en espacio normalizado)
    """
    if len(points) < 3:
        # fallback: repetir índices (evitar crash)
        if not points:
            return 0, 0, 0
        return 0, min(1, len(points)-1), min(2, len(points)-1)

    arr = np.array(points, dtype=float)
    i_mk = int(np.argmin(arr[:, 0]))
    i_en = int(np.argmin(arr[:, 1]))

    a = arr[i_mk]
    b = arr[i_en]
    # normalización min-max
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    denom = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
    norm = (arr - mins) / denom
    a_n = (a - mins) / denom
    b_n = (b - mins) / denom

    # distancia punto-recta (segmento) en 2D
    ab = b_n - a_n
    ab_norm2 = float(np.dot(ab, ab))
    if ab_norm2 == 0.0:
        # extremos iguales: rodilla = punto más alejado del extremo
        d = np.linalg.norm(norm - a_n, axis=1)
        i_knee = int(np.argmax(d))
        return i_mk, i_en, i_knee

    # proyección al segmento
    t = np.clip(((norm - a_n) @ ab) / ab_norm2, 0.0, 1.0)
    proj = a_n + t[:, None] * ab
    dist = np.linalg.norm(norm - proj, axis=1)

    # evitar seleccionar mismo índice que extremos
    dist[i_mk] = -1.0
    dist[i_en] = -1.0
    i_knee = int(np.argmax(dist))
    return i_mk, i_en, i_knee


def save_op_table(out_dir: Path, scenario_name: str, policy: str, label: str, records) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"op_table_{scenario_name}_{policy}_{label}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["job", "operation", "machine", "start", "end", "duration", "energy"])
        for r in records:
            w.writerow([r.job_id, r.op_id, r.machine, float(r.start), float(r.end), float(r.end - r.start), float(r.energy)])
    return path


def plot_gantt(out_dir: Path, scenario_name: str, policy: str, label: str, records, num_machines: int) -> Path:
    """Gantt simple: filas = máquinas, barras = operaciones."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"gantt_{scenario_name}_{policy}_{label}.png"

    plt.figure(figsize=(12, 5))

    # Agrupar por máquina
    by_m: Dict[int, List] = {m: [] for m in range(1, num_machines + 1)}
    for r in records:
        by_m[r.machine].append(r)

    # Ordenar por inicio para dibujar
    for m in by_m:
        by_m[m].sort(key=lambda x: x.start)

    for m in range(1, num_machines + 1):
        y = m
        for r in by_m[m]:
            plt.barh(y=y, width=(r.end - r.start), left=r.start)
            plt.text(r.start + 0.01, y, f"J{r.job_id}-O{r.op_id}", va="center", fontsize=7)

    plt.yticks(range(1, num_machines + 1), [f"M{m}" for m in range(1, num_machines + 1)])
    plt.xlabel("Tiempo")
    plt.ylabel("Máquina")
    plt.title(f"Gantt — {scenario_name} — {policy} — {label}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path

def save_hv_stats(out_dir: Path, scenario_name: str, hv_stats: Dict[int, Dict[str, List[float]]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV por escenario
    csv_path = out_dir / f"hv_{scenario_name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["generation", "policy", "min", "max", "mean", "std"])
        for gen in sorted(hv_stats.keys()):
            for pol in POLICIES:
                vals = hv_stats[gen][pol]
                if not vals:
                    continue
                w.writerow([gen, pol, np.min(vals), np.max(vals), np.mean(vals), np.std(vals)])

def plot_hv_evolution(out_dir: Path, scenario_name: str, hv_stats: Dict[int, Dict[str, List[float]]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    gens = sorted(hv_stats.keys())
    plt.figure(figsize=(10, 6))
    for pol in POLICIES:
        means = []
        for g in gens:
            vals = hv_stats[g][pol]
            means.append(float(np.mean(vals)) if vals else 0.0)
        plt.plot(gens, means, marker="o", label=pol)
    plt.title(f"Evolución del Hipervolumen — {scenario_name}")
    plt.xlabel("Generación")
    plt.ylabel("HV promedio")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / f"hv_evolution_{scenario_name}.png", dpi=150)
    plt.close()


def save_reference_points(out_dir: Path, scenario_name: str, ref_by_policy: Dict[str, Tuple[float, float]]) -> None:
    """Guarda los puntos de referencia usados para HV (uno por política)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"hv_ref_{scenario_name}.json"
    data = {pol: {"ref_makespan": float(ref[0]), "ref_energy": float(ref[1])} for pol, ref in ref_by_policy.items()}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def save_seed_hv_curve(
    seed_dir: Path,
    hv_generations: List[int],
    policies: List[str],
    hv_by_gen_policy: Dict[int, Dict[str, float]],
) -> Path:
    """
    Guarda la curva HV de UNA semilla (seed) para gen 20/40/60/80/100 y todas las políticas.
    Estructura:
      gen,FIFO,LTP,STP,RRFIFO,RRLTP,RRECA
    """
    seed_dir.mkdir(parents=True, exist_ok=True)
    path = seed_dir / "hv_curve.csv"

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gen"] + policies)
        for g in hv_generations:
            row = [g]
            for p in policies:
                row.append(float(hv_by_gen_policy.get(g, {}).get(p, 0.0)))
            w.writerow(row)

    return path


def save_seed_snapshots_json(seed_dir: Path, snapshots: Dict[int, Dict[str, List[Tuple[float, float]]]], hv_generations: List[int]) -> Path:
    """
    Guarda snapshots (solo gens de interés) para UNA seed.
    snapshots[g][policy] = [(mk,en), ...] del Frente 0.
    """
    seed_dir.mkdir(parents=True, exist_ok=True)
    path = seed_dir / "snapshots_hv.json"

    # Convertimos llaves int a str para JSON
    trimmed = {str(g): snapshots.get(g, {}) for g in hv_generations}
    path.write_text(json.dumps(trimmed, indent=2), encoding="utf-8")
    return path


def plot_hv_boxplot_gen100(out_dir: Path, scenario_name: str, hv_stats: Dict[int, Dict[str, List[float]]], policies: List[str]) -> Path:
    """
    Boxplot de HV (Gen 100) por política para mostrar variabilidad entre seeds (30 runs).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"hv_boxplot_gen100_{scenario_name}.png"

    data = []
    labels = []
    gen = 100
    for p in policies:
        vals = hv_stats.get(gen, {}).get(p, [])
        if vals:
            data.append(vals)
            labels.append(p)

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.title(f"Distribución del Hipervolumen (Gen 100) — {scenario_name}")
    plt.xlabel("Política")
    plt.ylabel("HV (30 ejecuciones)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path

