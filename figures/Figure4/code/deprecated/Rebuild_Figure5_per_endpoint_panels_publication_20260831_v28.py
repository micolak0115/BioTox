#!/usr/bin/env python3
"""Figure 5 v28: panel-E-aligned endpoints and taller gene axes.

This plotting-only revision preserves v27 typography and statistical semantics.
Endpoint-label centers are aligned to the measured left edge of panel E's
``NR-Aromatase`` label, and the gene axes are enlarged vertically without
changing the height or vertical placement of the paired GSEA axes.
"""
from __future__ import annotations

from pathlib import Path

import Rebuild_Figure5_per_endpoint_panels_publication_20260831_v27 as publication


base = publication.base
base.OUT_DIR = Path(
    "/home/kyungan/scripts/BioTox/publication/final/figures_revised/Figure5/"
    "unadjusted_global_zbeta_exploratory_q010_publication_20260831_v28"
)
base.FIG_STEM = "Figure5_per_endpoint_panels_A4_q010_publication_20260831_v28"

# Measured from the v27 renderer: moving from -0.465 to -0.45302 aligns the
# endpoint-label center with the left edge of panel E's NR-Aromatase label.
base.ENDPOINT_LABEL_X = -0.45302

# Give the coefficient bars more vertical breathing room while keeping each
# GSEA axis at its established size and position.
base.GENE_HEIGHT_SCALE = 1.28
base.GENE_VERTICAL_SHIFT = 0.0


if __name__ == "__main__":
    base.main()
