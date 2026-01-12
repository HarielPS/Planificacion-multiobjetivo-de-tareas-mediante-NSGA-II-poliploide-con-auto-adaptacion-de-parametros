from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

from .individual import Individual


def uniform_polyploid_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    p_crossover: float,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Cruza uniforme aplicada a TODOS los cromosomas (matriz 6 x L).

    Con prob p_crossover, para cada gen se intercambia entre padres.
    """
    a = parent_a.copy()
    b = parent_b.copy()
    mask = rng.random(size=a.shape) < p_crossover
    child1 = np.where(mask, b, a)
    child2 = np.where(mask, a, b)
    return child1.astype(int), child2.astype(int)


def mutate_inter_chromosome(chroms: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Mutación inter-cromosoma: intercambio de cromosomas completos."""
    out = chroms.copy()
    if rng.random() >= p:
        return out
    c = out.shape[0]
    i, j = rng.integers(0, c, size=2)
    out[[i, j], :] = out[[j, i], :]
    return out


def mutate_swap_genes(chroms: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Intercambio recíproco: swap de genes dentro de un cromosoma."""
    out = chroms.copy()
    if rng.random() >= p:
        return out
    c, L = out.shape
    chrom_idx = int(rng.integers(0, c))
    i, j = rng.integers(0, L, size=2)
    out[chrom_idx, i], out[chrom_idx, j] = out[chrom_idx, j], out[chrom_idx, i]
    return out


def mutate_shift_segment(chroms: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Desplazamiento: rotación circular de un segmento dentro de un cromosoma."""
    out = chroms.copy()
    if rng.random() >= p:
        return out
    c, L = out.shape
    chrom_idx = int(rng.integers(0, c))
    i, j = sorted(rng.integers(0, L, size=2))
    if j - i < 2:
        return out
    seg = out[chrom_idx, i:j].copy()
    out[chrom_idx, i:j] = np.roll(seg, 1)
    return out


def apply_mutations(
    ind: Individual,
    p_total: float,
    weights: Tuple[float, float, float],
    rng: np.random.Generator
) -> Individual:
    """Aplica las 3 mutaciones repartiendo el presupuesto p_total.

    weights = (w1,w2,w3) y w1+w2+w3 = 1.0 (idealmente).
    """
    w1, w2, w3 = weights
    s = float(w1 + w2 + w3) if (w1 + w2 + w3) > 0 else 1.0
    p1 = p_total * (w1 / s)
    p2 = p_total * (w2 / s)
    p3 = p_total * (w3 / s)

    chroms = ind.chromosomes
    chroms = mutate_inter_chromosome(chroms, p1, rng)
    chroms = mutate_swap_genes(chroms, p2, rng)
    chroms = mutate_shift_segment(chroms, p3, rng)

    ind.chromosomes = chroms.astype(int)
    return ind
