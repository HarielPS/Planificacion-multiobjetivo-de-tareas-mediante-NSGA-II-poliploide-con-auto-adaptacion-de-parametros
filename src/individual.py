from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

from .config import POLICIES
from .io_scenario import Scenario
from .policies import OpKey
from .schedule_decode import decode_and_simulate, OpRecord

Fitness = Tuple[float, float]

@dataclass
class Individual:
    chromosomes: np.ndarray  # shape (6, total_ops)
    fitness: List[Fitness] = field(default_factory=list)          # por política
    records: List[List[OpRecord]] = field(default_factory=list)   # por política (para Gantt/output)
    ranks: np.ndarray = field(default_factory=lambda: np.zeros(len(POLICIES), dtype=int))
    crowd: np.ndarray = field(default_factory=lambda: np.zeros(len(POLICIES), dtype=float))

    def evaluate(self, scn: Scenario, orders: Dict[str, List[OpKey]]) -> None:
        self.fitness = []
        self.records = []
        for pi, pol in enumerate(POLICIES):
            mk, en, recs = decode_and_simulate(scn, orders[pol], self.chromosomes[pi])
            # fitness en el orden (makespan, energy) para dominancia
            self.fitness.append((mk, en))
            self.records.append(recs)
