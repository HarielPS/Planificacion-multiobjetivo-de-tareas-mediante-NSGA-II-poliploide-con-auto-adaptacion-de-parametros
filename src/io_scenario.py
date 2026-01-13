from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import numpy as np


@dataclass
class Scenario:
    T: np.ndarray                 # tiempos: (num_operaciones, num_maquinas)
    E: np.ndarray                 # energía: (num_operaciones, num_maquinas)
    jobs: Dict[int, List[int]]    # job_id -> lista de operaciones

    @property
    def num_machines(self) -> int:
        return int(self.T.shape[1])

    @property
    def num_operations(self) -> int:
        return int(self.T.shape[0])

    @property
    def total_job_ops(self) -> int:
        return int(sum(len(v) for v in self.jobs.values()))


def load_scenario(path_txt: str | Path) -> Scenario:
    """Carga un escenario en el formato:
    #tiempos...
    <filas de tiempos>
    #consumo...
    <filas de energía>
    #Trabajos
    J1={O1,O2,...}
    ...

    Mantiene el orden de operaciones dentro de cada trabajo (para precedencias).
    """
    path = Path(path_txt)
    lines = path.read_text(encoding="utf-8").splitlines()

    T: List[List[float]] = []
    E: List[List[float]] = []
    jobs: Dict[int, List[int]] = {}

    section = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line.startswith("#tiempos"):
            section = "T"
            continue
        if line.startswith("#consumo"):
            section = "E"
            continue
        if line.startswith("#Trabajos"):
            section = "J"
            continue

        if section == "T":
            T.append([float(x) for x in line.split()])
        elif section == "E":
            E.append([float(x) for x in line.split()])
        elif section == "J":
            # J6={O1,O2,O4,O5}
            job_str, rhs = line.split("=", 1)
            job_id = int(job_str.strip().replace("J", ""))

            rhs = rhs.strip().replace("{", "").replace("}", "")
            tokens = [t.strip() for t in rhs.split(",") if t.strip()]
            ops: List[int] = []
            for token in tokens:
                ops.append(int(token.replace("O", "")))
            jobs[job_id] = ops

    T_arr = np.array(T, dtype=float)
    E_arr = np.array(E, dtype=float)

    # Validaciones básicas
    if T_arr.ndim != 2 or E_arr.ndim != 2:
        raise ValueError("Formato inválido: T/E deben ser matrices 2D.")
    if T_arr.shape != E_arr.shape:
        raise ValueError(f"Dimensiones incompatibles: T{T_arr.shape} vs E{E_arr.shape}")
    if not jobs:
        raise ValueError("No se encontraron trabajos (#Trabajos).")

    return Scenario(T=T_arr, E=E_arr, jobs=jobs)
