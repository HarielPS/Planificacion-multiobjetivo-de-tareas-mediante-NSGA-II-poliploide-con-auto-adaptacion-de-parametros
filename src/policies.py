from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from .config import POLICIES
from .io_scenario import Scenario

OpKey = Tuple[int, int]  # (job_id, op_id)

def _avg_by_operation(values: np.ndarray) -> np.ndarray:
    # values: (num_ops, num_machines) -> avg por operación (num_ops,)
    return values.mean(axis=1)

def build_orders(scn: Scenario) -> Dict[str, List[OpKey]]:
    """Antes de iniciar el proceso evolutivo, se calcula el orden de atención por política.
    Regresa dict: policy_name -> lista de (job_id, op_id) del tamaño total_job_ops.
    """
    if not scn.jobs:
        raise ValueError("No hay trabajos en el escenario.")

    # Promedios por operación global (para LTP/STP/RRECA)
    avg_time = _avg_by_operation(scn.T)   # index op-1
    avg_energy = _avg_by_operation(scn.E) # index op-1

    # 1) FIFO: jobs en orden de id, ops en el orden del archivo
    fifo: List[OpKey] = []
    for job_id in sorted(scn.jobs.keys()):
        for op in scn.jobs[job_id]:
            fifo.append((job_id, op))

    # 2) LTP: ordenar TODAS las (job,op) por tiempo promedio descendente
    ltp = sorted(fifo, key=lambda k: (avg_time[k[1]-1], k[0]), reverse=True)

    # 3) STP: ordenar por tiempo promedio ascendente
    stp = sorted(fifo, key=lambda k: (avg_time[k[1]-1], k[0]))

    # 4) RRFIFO: round-robin por jobs, tomando la siguiente operación de cada job
    rrfifo: List[OpKey] = []
    pending = {j: ops.copy() for j, ops in scn.jobs.items()}
    total = scn.total_job_ops
    while len(rrfifo) < total:
        for job_id in sorted(pending.keys()):
            if pending[job_id]:
                rrfifo.append((job_id, pending[job_id].pop(0)))
                if len(rrfifo) >= total:
                    break

    # 5) RRLTP: round-robin, pero el orden de jobs se define por "carga" (suma de promedios de sus ops) desc
    def job_load(job_id: int) -> float:
        return float(sum(avg_time[op-1] for op in scn.jobs[job_id]))

    rrltp: List[OpKey] = []
    pending2 = {j: ops.copy() for j, ops in scn.jobs.items()}
    job_order = sorted(pending2.keys(), key=lambda j: job_load(j), reverse=True)
    while len(rrltp) < total:
        for job_id in job_order:
            if pending2[job_id]:
                rrltp.append((job_id, pending2[job_id].pop(0)))
                if len(rrltp) >= total:
                    break

    # 6) RRECA: ordenar TODAS las (job,op) por energía promedio descendente
    rreca = sorted(fifo, key=lambda k: (avg_energy[k[1]-1], k[0]), reverse=True)

    orders = {
        "FIFO": fifo,
        "LTP": ltp,
        "STP": stp,
        "RRFIFO": rrfifo,
        "RRLTP": rrltp,
        "RRECA": rreca,
    }

    # Consistencia
    for name in POLICIES:
        if name not in orders:
            raise ValueError(f"Falta política: {name}")
        if len(orders[name]) != total:
            raise ValueError(f"Orden {name} inválido: {len(orders[name])} != {total}")

    return orders
