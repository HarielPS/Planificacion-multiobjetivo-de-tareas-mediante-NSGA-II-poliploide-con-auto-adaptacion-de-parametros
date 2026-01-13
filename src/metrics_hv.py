from __future__ import annotations
from typing import List, Tuple
import numpy as np

Fitness = Tuple[float, float]

def hypervolume(points: List[Fitness], ref: Tuple[float, float]) -> float:
    """Hipervolumen de un conjunto de puntos respecto a un punto de referencia (usando pymoo)
    """
    if not points:
        return 0.0

    pts = np.array(points, dtype=float)

    try:
        from pymoo.indicators.hv import HV
        hv = HV(ref_point=np.array(ref, dtype=float))
        return float(hv.do(pts))
    except Exception:
        # ordenar por makespan ascendente
        pts = pts[np.argsort(pts[:, 0])]
        hv_val = 0.0
        prev_y = ref[1]
        for x, y in pts:
            width = max(ref[0] - x, 0.0)
            height = max(prev_y - y, 0.0)
            hv_val += width * height
            prev_y = min(prev_y, y)
        return float(hv_val)

def choose_reference(points: List[Fitness], scale: float = 1.2) -> Tuple[float, float]:
    pts = np.array(points, dtype=float)
    mx = float(pts[:, 0].max()) * scale
    my = float(pts[:, 1].max()) * scale
    return (mx, my)
