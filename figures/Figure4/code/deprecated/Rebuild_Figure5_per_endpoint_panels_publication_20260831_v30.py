#!/usr/bin/env python3
"""Figure 5 v30: endpoint labels aligned to the start of Aromatase.

This plotting-only revision preserves v29 while shifting every A-D endpoint
label rightward so its center aligns with the rendered start of ``Aromatase``
in panel E's ``NR-Aromatase`` y-tick label.
"""
from __future__ import annotations

from pathlib import Path

import Rebuild_Figure5_per_endpoint_panels_publication_20260831_v29 as publication


base = publication.base
base.OUT_DIR = Path(
    "/home/kyungan/scripts/BioTox/publication/final/figures_revised/Figure5/"
    "unadjusted_global_zbeta_exploratory_q010_publication_20260831_v30"
)
base.FIG_STEM = "Figure5_per_endpoint_panels_A4_q010_publication_20260831_v30"

# Renderer-derived position: endpoint center = left edge of the ``Aromatase``
# substring, leaving an explicit gap before the nearest gene tick label.
base.ENDPOINT_LABEL_X = -0.338757


if __name__ == "__main__":
    base.main()
