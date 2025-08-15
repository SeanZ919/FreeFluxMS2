
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple
import numpy as np

def popcount(x: int) -> int:
    return x.bit_count() if hasattr(int, "bit_count") else bin(x).count("1")

def isotopomer_to_mid(p_iso: np.ndarray, n_atoms: int) -> np.ndarray:
    mid = np.zeros(n_atoms + 1, dtype=float)
    for i, p in enumerate(p_iso):
        mid[popcount(i)] += p
    s = mid.sum()
    if s > 0:
        mid /= s
    return mid

def marginalize_isotopomer(p_iso_prec: np.ndarray, keep: Sequence[int], n_prec_atoms: int) -> np.ndarray:
    keep = list(keep)
    k = len(keep)
    p_iso_prod = np.zeros(2 ** k, dtype=float)
    for prec_state in range(2 ** n_prec_atoms):
        p = p_iso_prec[prec_state]
        if p == 0.0:
            continue
        prod_state = 0
        for out_pos, atom_idx in enumerate(keep):
            bit = (prec_state >> atom_idx) & 1
            prod_state |= (bit << out_pos)
        p_iso_prod[prod_state] += p
    s = float(p_iso_prod.sum())
    if s > 0:
        p_iso_prod /= s
    return p_iso_prod

@dataclass(frozen=True)
class FragmentationOp:
    channels: Tuple[Tuple[Tuple[int, ...], float], ...]
    n_prec_atoms: int

    @staticmethod
    def single_channel(keep_indices: Sequence[int], n_prec_atoms: int, weight: float = 1.0) -> "FragmentationOp":
        return FragmentationOp(channels=((tuple(sorted(keep_indices)), float(weight)),), n_prec_atoms=int(n_prec_atoms))

    def simulate_from_isotopomer(self, p_iso_prec: np.ndarray):
        assert len(p_iso_prec) == 2 ** self.n_prec_atoms, "precursor isotopomer length mismatch"
        p_iso_acc = None
        for keep, w in self.channels:
            p_iso_prod = marginalize_isotopomer(p_iso_prec, keep, self.n_prec_atoms)
            p_iso_acc = p_iso_prod * w if p_iso_acc is None else p_iso_acc + p_iso_prod * w
        if p_iso_acc is None:
            raise ValueError("No channels defined")
        sw = float(np.sum([w for _, w in self.channels]))
        if sw > 0:
            p_iso_acc = p_iso_acc / sw
        mid = isotopomer_to_mid(p_iso_acc, len(self.channels[0][0]))
        return p_iso_acc, mid

def simulate_ms2_from_isotopomer(p_iso_prec: np.ndarray, keep_indices: Sequence[int], n_prec_atoms: int) -> np.ndarray:
    op = FragmentationOp.single_channel(keep_indices, n_prec_atoms)
    _, mid = op.simulate_from_isotopomer(p_iso_prec)
    return mid
