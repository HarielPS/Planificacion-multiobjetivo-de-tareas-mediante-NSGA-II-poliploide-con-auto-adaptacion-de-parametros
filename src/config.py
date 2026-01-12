from dataclasses import dataclass
from typing import List, Tuple

POLICIES: List[str] = ["FIFO", "LTP", "STP", "RRFIFO", "RRLTP", "RRECA"]
POLICY_GROUP_1 = ["FIFO", "LTP", "STP"]
POLICY_GROUP_2 = ["RRFIFO", "RRLTP", "RRECA"]

# Auto-adaptación (valores sugeridos por el documento del proyecto)
CROSSOVER_LEVELS: List[float] = [0.6, 0.8, 0.9]
MUTATION_LEVELS: List[float]  = [0.01, 0.1, 0.2]

# Puntos donde se calcula HV (metodología de la práctica 2)
HV_GENERATIONS: List[int] = [20, 40, 60, 80, 100]

@dataclass(frozen=True)
class GAConfig:
    population_size: int = 10
    generations: int = 100

    # Valores iniciales (práctica 2 sugiere cruza 0.8 y varias mutaciones)
    crossover_init_idx: int = 1  # 0.8
    mutation_init_idx: int = 1   # 0.1

    # Distribución interna del presupuesto de mutación entre 3 operadores
    # (mantenemos suma = 1.0 para repartir pm de forma consistente)
    mut_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)

    # Umbral de éxito: % de cromosomas-hijo que caen en frente 0
    success_front0_threshold: float = 0.25
