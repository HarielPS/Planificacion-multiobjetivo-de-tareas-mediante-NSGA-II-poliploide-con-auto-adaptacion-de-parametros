from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from .io_scenario import Scenario
from .policies import OpKey

@dataclass
class OpRecord:
    job_id: int
    op_id: int
    machine: int
    start: float
    end: float
    energy: float

def decode_and_simulate(
    scn: Scenario,
    order: List[OpKey],
    chromosome: np.ndarray
) -> Tuple[float, float, List[OpRecord]]:
    """Decodifica un cromosoma (enteros 1..num_machines) con un orden de atención (policy),
    simula ejecución con restricciones de precedencia y disponibilidad de máquinas.

    Regresa:
      makespan, total_energy, registros_por_operacion
    """
    num_machines = scn.num_machines
    if chromosome.shape[0] != len(order):
        raise ValueError("Cromosoma y orden no coinciden en tamaño.")

    # disponibilidad de máquina
    mach_ready = np.zeros(num_machines + 1, dtype=float)  # index 1..p
    # fin de la última operación por trabajo
    job_last_end: Dict[int, float] = {j: 0.0 for j in scn.jobs.keys()}

    records: List[OpRecord] = []
    total_energy = 0.0

    for idx, (job_id, op_id) in enumerate(order):
        m = int(chromosome[idx])
        if m < 1 or m > num_machines:
            m = max(1, min(num_machines, m))

        # precedencia: no inicia antes del fin del job
        earliest = job_last_end.get(job_id, 0.0)
        # disponibilidad de máquina
        start = max(earliest, mach_ready[m])

        dur = float(scn.T[op_id - 1, m - 1])
        energy = float(scn.E[op_id - 1, m - 1])

        end = start + dur
        mach_ready[m] = end
        job_last_end[job_id] = end

        total_energy += energy
        records.append(OpRecord(job_id=job_id, op_id=op_id, machine=m, start=start, end=end, energy=energy))

    makespan = float(mach_ready.max())
    return makespan, float(total_energy), records
