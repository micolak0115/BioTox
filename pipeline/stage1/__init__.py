"""Stage 1 molecular-prediction and ensemble-construction package.

The legacy ``chem`` package alias is retained at runtime solely so archived
pickles that record ``chem.splitter.DataSplit`` remain readable after the
public package rename to :mod:`stage1`.
"""
from __future__ import annotations

import sys


sys.modules.setdefault("chem", sys.modules[__name__])
