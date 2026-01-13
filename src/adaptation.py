from __future__ import annotations

from typing import Tuple
import numpy as np

from .config import CROSSOVER_LEVELS, MUTATION_LEVELS


def measure_success(children_ranks: np.ndarray, threshold: float) -> Tuple[float, bool]:
    """Mide 'éxito' para auto-adaptación.

    children_ranks: ndarray shape (num_children, 6) con ranks por política (0 = Frente 0).
    Éxito = (% de cromosomas-hijo en Frente 0) >= threshold.
    """
    if children_ranks.size == 0:
        return 0.0, False

    total = int(children_ranks.size)
    good = int((children_ranks == 0).sum())
    rate = good / max(total, 1)
    return float(rate), (rate >= threshold)


def adapt_indices(idx_c: int, idx_m: int, success: bool) -> Tuple[int, int]:
    """Ajuste de índices de pc/pm

    - Si NO hay éxito: subimos exploración => subir mutación, si ya está al máximo, subir cruza.
    - Si SÍ hay éxito: bajamos mutación (explotación) y acercamos cruza al valor medio (0.8).
    """
    if not success:
        if idx_m < len(MUTATION_LEVELS) - 1:
            idx_m += 1
        elif idx_c < len(CROSSOVER_LEVELS) - 1:
            idx_c += 1
    else:
        if idx_m > 0:
            idx_m -= 1
        if idx_c > 1:
            idx_c -= 1
        elif idx_c < 1:
            idx_c += 1
    return idx_c, idx_m
