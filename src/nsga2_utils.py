from __future__ import annotations
from typing import List, Tuple
import numpy as np

Fitness = Tuple[float, float]  # (makespan, energy)

def dominates(a: Fitness, b: Fitness) -> bool:
    # minimización: a domina a b si no es peor en ninguna y mejor en al menos una
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

def fast_non_dominated_sort(points: List[Fitness]) -> List[List[int]]:
    n = len(points)
    S = [[] for _ in range(n)]
    n_dom = [0] * n
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(points[p], points[q]):
                S[p].append(q)
            elif dominates(points[q], points[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    if not fronts[-1]:
        fronts.pop()
    return fronts

def crowding_distance(points: List[Fitness], idxs: List[int]) -> np.ndarray:
    # distancias solo para los índices del frente
    if len(idxs) == 0:
        return np.array([])
    if len(idxs) <= 2:
        return np.full(len(idxs), np.inf)

    front_pts = np.array([points[i] for i in idxs], dtype=float)  # shape (k,2)
    k = front_pts.shape[0]
    dist = np.zeros(k, dtype=float)

    for obj in range(2):
        order = np.argsort(front_pts[:, obj])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        minv = front_pts[order[0], obj]
        maxv = front_pts[order[-1], obj]
        denom = max(maxv - minv, 1e-12)
        for t in range(1, k - 1):
            dist[order[t]] += (front_pts[order[t + 1], obj] - front_pts[order[t - 1], obj]) / denom

    return dist

def tournament_select(pop_size: int, ranks: np.ndarray, crowd: np.ndarray, rng: np.random.Generator) -> int:
    # torneo binario (min rank, max crowd)
    a = int(rng.integers(0, pop_size))
    b = int(rng.integers(0, pop_size))
    if ranks[a] < ranks[b]:
        return a
    if ranks[b] < ranks[a]:
        return b
    # desempate por crowding (más grande = mejor)
    return a if crowd[a] >= crowd[b] else b
