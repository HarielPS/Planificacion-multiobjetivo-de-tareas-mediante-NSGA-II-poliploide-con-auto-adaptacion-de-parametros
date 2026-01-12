from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from .config import GAConfig, POLICIES, CROSSOVER_LEVELS, MUTATION_LEVELS, HV_GENERATIONS
from .io_scenario import Scenario
from .policies import build_orders
from .individual import Individual
from .nsga2_utils import fast_non_dominated_sort, crowding_distance, tournament_select
from .operators import uniform_polyploid_crossover, apply_mutations
from .adaptation import measure_success, adapt_indices

Fitness = Tuple[float, float]


@dataclass
class RunLog:
    pc_history: List[float]
    pm_history: List[float]
    success_rate_history: List[float]
    # snapshots[gen][policy] = lista de puntos (makespan, energia) del Frente 0 para esa política
    snapshots: Dict[int, Dict[str, List[Fitness]]]


def _evaluate_population(pop: List[Individual], scn: Scenario, orders) -> None:
    for ind in pop:
        ind.evaluate(scn, orders)


def _assign_rank_and_crowd_per_policy(pop: List[Individual]) -> None:
    """Asigna (rank ND) y (crowding) por política, como en la práctica 2.

    Cada individuo tiene 6 cromosomas => 6 evaluaciones => 6 frentes / crowding.
    """
    for pi, _ in enumerate(POLICIES):
        pts = [ind.fitness[pi] for ind in pop]
        fronts = fast_non_dominated_sort(pts)

        ranks = np.empty(len(pop), dtype=int)
        crowd = np.zeros(len(pop), dtype=float)

        for r, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = r
            cd = crowding_distance(pts, front)
            for j, idx in enumerate(front):
                crowd[idx] = cd[j]

        for i, ind in enumerate(pop):
            ind.ranks[pi] = int(ranks[i])
            ind.crowd[pi] = float(crowd[i])


def _front0_points(pop: List[Individual]) -> Dict[str, List[Fitness]]:
    """Devuelve los puntos (mk,en) del Frente 0 por política."""
    out: Dict[str, List[Fitness]] = {p: [] for p in POLICIES}
    for pi, pol in enumerate(POLICIES):
        out[pol] = [ind.fitness[pi] for ind in pop if int(ind.ranks[pi]) == 0]
    return out


def run_nsga2(
    scn: Scenario,
    cfg: GAConfig,
    seed: int,
    snapshot_gens: Optional[List[int]] = None,
) -> Tuple[List[Individual], RunLog]:
    """Ejecuta una corrida de NSGA-II poliploide (multipolítica).

    - Evalúa por política (6 frentes / crowding).
    - Auto-adapta pc y pm usando los conjuntos del proyecto.
    - Toma snapshots del Frente 0 por política en generaciones HV_GENERATIONS (por defecto).
    """
    if snapshot_gens is None:
        snapshot_gens = HV_GENERATIONS

    rng = np.random.default_rng(seed)
    orders = build_orders(scn)

    L = scn.total_job_ops
    p = scn.num_machines

    # Población inicial: 6 x L con enteros 1..p
    pop: List[Individual] = []
    for _ in range(cfg.population_size):
        chrom = rng.integers(1, p + 1, size=(len(POLICIES), L), endpoint=False)
        pop.append(Individual(chromosomes=chrom))

    _evaluate_population(pop, scn, orders)
    _assign_rank_and_crowd_per_policy(pop)

    pc_idx = cfg.crossover_init_idx
    pm_idx = cfg.mutation_init_idx

    pc_hist: List[float] = []
    pm_hist: List[float] = []
    succ_hist: List[float] = []
    snapshots: Dict[int, Dict[str, List[Fitness]]] = {}

    for gen in range(1, cfg.generations + 1):
        pc = float(CROSSOVER_LEVELS[pc_idx])
        pm = float(MUTATION_LEVELS[pm_idx])
        pc_hist.append(pc)
        pm_hist.append(pm)

        children: List[Individual] = []

        while len(children) < cfg.population_size:
            ranks_mean_parent = np.mean([ind.ranks for ind in pop], axis=1)
            crowd_mean_parent = np.mean([ind.crowd for ind in pop], axis=1)
            i1 = tournament_select(len(pop), ranks_mean_parent, crowd_mean_parent, rng)
            i2 = tournament_select(len(pop), ranks_mean_parent, crowd_mean_parent, rng)
            p1 = pop[i1]
            p2 = pop[i2]

            c1_chrom, c2_chrom = uniform_polyploid_crossover(p1.chromosomes, p2.chromosomes, pc, rng)
            c1 = Individual(chromosomes=c1_chrom)
            c2 = Individual(chromosomes=c2_chrom)

            # Mutaciones (reparto del presupuesto pm en 3 operadores)
            c1 = apply_mutations(c1, pm, cfg.mut_weights, rng)
            c2 = apply_mutations(c2, pm, cfg.mut_weights, rng)

            children.append(c1)
            if len(children) < cfg.population_size:
                children.append(c2)

        _evaluate_population(children, scn, orders)

        combined = pop + children
        _assign_rank_and_crowd_per_policy(combined)

        # Éxito: % de cromosomas-hijo que caen en frente 0 (en la población combinada)
        children_ranks = np.array([c.ranks for c in children], dtype=int)
        succ_rate, succ_bool = measure_success(children_ranks, cfg.success_front0_threshold)
        succ_hist.append(float(succ_rate))
        pc_idx, pm_idx = adapt_indices(pc_idx, pm_idx, succ_bool)

        # Selección de sobrevivientes:
        # (Idealmente en P2 hay "intercambio cromosómico"; aquí usamos un criterio global coherente
        # con el enfoque multi-política: promedio rank asc y promedio crowd desc.)
        ranks_mean = np.mean([ind.ranks for ind in combined], axis=1)
        crowd_mean = np.mean([ind.crowd for ind in combined], axis=1)
        order_idx = np.lexsort((-crowd_mean, ranks_mean))

        pop = [combined[i] for i in order_idx[:cfg.population_size]]
        _assign_rank_and_crowd_per_policy(pop)

        # Snapshot HV en generaciones específicas
        if gen in snapshot_gens:
            snapshots[gen] = _front0_points(pop)

    return pop, RunLog(pc_history=pc_hist, pm_history=pm_hist, success_rate_history=succ_hist, snapshots=snapshots)
