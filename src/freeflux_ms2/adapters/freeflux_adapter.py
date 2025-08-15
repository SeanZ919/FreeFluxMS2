
from typing import Sequence, Tuple
import numpy as np
from ..fragmentation import FragmentationOp

def get_precursor_isotopomer_from_freeflux(model, met: str, atoms_1based: Sequence[int]) -> Tuple[np.ndarray, int]:
    raise NotImplementedError

def simulate_ms2_from_freeflux(model, met: str, precursor_atoms_1based: Sequence[int], product_atoms_1based: Sequence[int]):
    p_iso_prec, n = get_precursor_isotopomer_from_freeflux(model, met, precursor_atoms_1based)
    keep0 = [a-1 for a in product_atoms_1based]
    op = FragmentationOp.single_channel(keep0, n)
    _, mid = op.simulate_from_isotopomer(p_iso_prec)
    return mid
