
import numpy as np
from freeflux_ms2.fragmentation import FragmentationOp, marginalize_isotopomer, isotopomer_to_mid

def demo():
    n = 4
    rng = np.random.default_rng(42)
    p_iso_prec = rng.random(2**n); p_iso_prec /= p_iso_prec.sum()
    keep0 = [1,2,3]
    op = FragmentationOp.single_channel(keep0, n)
    p_iso_prod, mid = op.simulate_from_isotopomer(p_iso_prec)
    print("Product MID:", np.round(mid, 4))
    mid_direct = isotopomer_to_mid(marginalize_isotopomer(p_iso_prec, keep0, n), len(keep0))
    assert np.allclose(mid, mid_direct), "Mismatch"
    print("Check passed.")

if __name__ == "__main__":
    demo()
