
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional

@dataclass
class ProductSpec:
    name: str
    precursor_met: str
    precursor_atoms: Sequence[int]  # 1-based in YAML
    product_atoms: Sequence[int]    # 1-based in YAML
    channels: Optional[List[Tuple[Sequence[int], float]]] = None
    isolation_purity: Optional[float] = None
    report: str = "MID"
