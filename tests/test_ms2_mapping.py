
import numpy as np
from freeflux_ms2.fragmentation import marginalize_isotopomer, isotopomer_to_mid, FragmentationOp

def test_marginalization_agrees_with_operator():
    n = 3
    keep = [0,2]
    rng = np.random.default_rng(0)
    p_iso_prec = rng.random(2**n); p_iso_prec /= p_iso_prec.sum()
    op = FragmentationOp.single_channel(keep, n)
    p_iso_prod, mid = op.simulate_from_isotopomer(p_iso_prec)
    p_iso_prod_direct = marginalize_isotopomer(p_iso_prec, keep, n)
    mid_direct = isotopomer_to_mid(p_iso_prod_direct, len(keep))
    assert np.allclose(p_iso_prod, p_iso_prod_direct)
    assert np.allclose(mid, mid_direct)
