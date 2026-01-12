from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .config import GAConfig, POLICIES, HV_GENERATIONS
from .io_scenario import load_scenario
from .nsga2_core import run_nsga2
from .metrics_hv import hypervolume, choose_reference
from .policies import build_orders
from .schedule_decode import decode_and_simulate
from .reporting import (
    save_hv_stats,
    plot_hv_evolution,
    save_reference_points,
    save_hv_tables,
    select_extremes_and_knee,
    save_op_table,
    plot_gantt,
    save_seed_hv_curve,
    save_seed_snapshots_json,
    plot_hv_boxplot_gen100,
)


Fitness = Tuple[float, float]


def run_experiment_for_scenario(scn_path: Path, runs: int, out_dir: Path) -> None:
    scn = load_scenario(scn_path)
    cfg = GAConfig()

    scenario_name = scn_path.stem
    scenario_out_dir = out_dir / scenario_name
    runs_dir = scenario_out_dir / "runs"
    summary_dir = scenario_out_dir / "summary"
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Guardamos snapshots por corrida para calcular un mismo punto de referencia por política
    run_snapshots: List[Dict[int, Dict[str, List[Fitness]]]] = []

    # Guardamos también la población final de cada corrida para:
    # - seleccionar soluciones (extremos + rodilla)
    # - generar Gantt + tabla por operación
    run_final_pops: List = []
    run_seeds: List[int] = []

    # Recolectar TODOS los puntos no dominados (Frente 0) por política a lo largo de runs y gens
    all_points_by_policy: Dict[str, List[Fitness]] = {p: [] for p in POLICIES}

    base_seed = 12345  # fijo para reproducibilidad del experimento
    for r in range(runs):
        seed = base_seed + r
        final_pop, log = run_nsga2(scn, cfg, seed=seed, snapshot_gens=HV_GENERATIONS)
        run_snapshots.append(log.snapshots)
        run_final_pops.append(final_pop)
        run_seeds.append(seed)

        for g in HV_GENERATIONS:
            snap = log.snapshots.get(g, {})
            for p in POLICIES:
                all_points_by_policy[p].extend(snap.get(p, []))

        print(f"[{scenario_name}] run {r+1}/{runs} listo (seed={seed}).")

    # 1) Unimos TODOS los puntos (todas las políticas) en un solo conjunto
    all_points_global: List[Fitness] = []
    for p in POLICIES:
        all_points_global.extend(all_points_by_policy[p])

    # 2) Elegimos UN solo ref global para TODO el escenario
    if all_points_global:
        ref_global: Fitness = choose_reference(all_points_global, scale=1.2)
    else:
        ref_global = (1.0, 1.0)

    # 3) Reutilizamos tu estructura actual: mismo ref para todas las políticas
    ref_by_policy: Dict[str, Fitness] = {p: ref_global for p in POLICIES}


    save_reference_points(summary_dir, scenario_name, ref_by_policy)

    # --- Guardado por seed (logs ligeros para trazabilidad) ---
    for run_idx, snapshots in enumerate(run_snapshots):
        seed = run_seeds[run_idx]
        seed_dir = runs_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # Guardar snapshots (solo gens relevantes)
        save_seed_snapshots_json(seed_dir, snapshots, HV_GENERATIONS)

        # Calcular y guardar la curva HV de esta seed (usando el mismo ref por política)
        hv_by_gen_policy: Dict[int, Dict[str, float]] = {g: {} for g in HV_GENERATIONS}
        for g in HV_GENERATIONS:
            for p in POLICIES:
                pts = snapshots.get(g, {}).get(p, [])
                hv_by_gen_policy[g][p] = float(hypervolume(pts, ref_by_policy[p]))

        save_seed_hv_curve(seed_dir, HV_GENERATIONS, POLICIES, hv_by_gen_policy)


    # hv_stats[gen][policy] = [hv_run1, hv_run2, ...]
    hv_stats: Dict[int, Dict[str, List[float]]] = {g: {p: [] for p in POLICIES} for g in HV_GENERATIONS}

    for snapshots in run_snapshots:
        for g in HV_GENERATIONS:
            for p in POLICIES:
                pts = snapshots.get(g, {}).get(p, [])
                hv_val = hypervolume(pts, ref_by_policy[p])
                hv_stats[g][p].append(float(hv_val))

    save_hv_stats(summary_dir, scenario_name, hv_stats)
    plot_hv_evolution(summary_dir, scenario_name, hv_stats)

    # Tablas estilo Práctica 2 (dos grupos de políticas)
    save_hv_tables(summary_dir, scenario_name, hv_stats)
    plot_hv_boxplot_gen100(summary_dir, scenario_name, hv_stats, POLICIES)
    # --- Selección de 3 soluciones del frente final + Gantt ---
    # Estrategia práctica para el reporte:
    #   Para cada política, buscamos la corrida con mejor HV en generación 100
    #   (y en esa corrida tomamos el Frente 0 final para elegir extremos + rodilla).
    orders = build_orders(scn)

    best_run_idx_by_policy: Dict[str, int] = {p: 0 for p in POLICIES}
    best_hv_by_policy: Dict[str, float] = {p: -1.0 for p in POLICIES}

    # calcular HV en gen 100 por corrida y política
    for run_idx, snapshots in enumerate(run_snapshots):
        for p in POLICIES:
            pts = snapshots.get(100, {}).get(p, [])
            hv_val = float(hypervolume(pts, ref_by_policy[p]))
            if hv_val > best_hv_by_policy[p]:
                best_hv_by_policy[p] = hv_val
                best_run_idx_by_policy[p] = run_idx

    for p in POLICIES:
        idx = best_run_idx_by_policy[p]
        pop = run_final_pops[idx]

        # Frente 0 final por política
        pi = POLICIES.index(p)
        front_inds = [ind for ind in pop if int(ind.ranks[pi]) == 0]
        if len(front_inds) == 0:
            continue

        points = [ind.fitness[pi] for ind in front_inds]
        i_mk, i_en, i_knee = select_extremes_and_knee(points)

        picks = [
            ("mk_min", front_inds[i_mk]),
            ("en_min", front_inds[i_en]),
            ("knee", front_inds[i_knee]),
        ]

        for label, ind in picks:
            chrom = ind.chromosomes[pi]
            mk, en, records = decode_and_simulate(scn, orders[p], chrom)

            # tabla por operación + gantt
            save_op_table(summary_dir, scenario_name, p, label, records)
            plot_gantt(summary_dir, scenario_name, p, label, records, num_machines=scn.num_machines)

        # registro de qué corrida se usó
        meta_path = summary_dir / f"selected_run_{scenario_name}_{p}.txt"
        meta_path.write_text(
            f"policy={p}\n"
            f"selected_run_index={idx}\n"
            f"seed={run_seeds[idx]}\n"
            f"hv_gen100={best_hv_by_policy[p]:.6f}\n",
            encoding="utf-8",
        )

    print(f"[{scenario_name}] listo. Resultados en: {summary_dir.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=str, required=True, help="Ruta a Escenario*.txt o 'all'")
    ap.add_argument("--runs", type=int, default=30, help="Ejecuciones independientes (>=30 recomendado)")
    ap.add_argument("--out", type=str, default="outputs", help="Directorio de salida")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.scenario.lower() == "all":
        for name in ["Escenario1.txt", "Escenario2.txt", "Escenario3.txt", "Escenario4.txt"]:
            scn_path = Path("data") / name
            if scn_path.exists():
                run_experiment_for_scenario(scn_path, args.runs, out_dir)
            else:
                print(f"[WARN] No encuentro {scn_path}. Copia tus escenarios a /data")
    else:
        scn_path = Path(args.scenario)
        if not scn_path.exists():
            # permitir pasar solo el nombre si está en /data
            alt = Path("data") / args.scenario
            if alt.exists():
                scn_path = alt
            else:
                raise FileNotFoundError(f"No encuentro el escenario: {args.scenario}")
        run_experiment_for_scenario(scn_path, args.runs, out_dir)


if __name__ == "__main__":
    main()
