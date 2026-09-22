#!/usr/bin/env python3
"""Figure 5 v31: concise colorbar annotation.

This plotting-only revision preserves v30 and changes only the displayed
colorbar annotation to ``Active fraction``. Statistical thresholds and audit
outputs remain unchanged.
"""
from __future__ import annotations

from pathlib import Path

import Rebuild_Figure5_per_endpoint_panels_publication_20260831_v30 as publication


base = publication.base
base.OUT_DIR = Path(
    "/home/kyungan/scripts/BioTox/publication/final/figures_revised/Figure5/"
    "unadjusted_global_zbeta_exploratory_q010_publication_20260831_v31"
)
base.FIG_STEM = "Figure5_per_endpoint_panels_A4_q010_publication_20260831_v31"
base.COLORBAR_LABEL = "Active fraction"


if __name__ == "__main__":
    base.main()
