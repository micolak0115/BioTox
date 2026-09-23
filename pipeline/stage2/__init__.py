"""Stage 2 transcriptomic residual-learning package.

The legacy ``bio`` alias keeps archived joblib/pickle references importable;
all current source code imports :mod:`stage2` directly.
"""
from __future__ import annotations

import sys


sys.modules.setdefault("bio", sys.modules[__name__])
