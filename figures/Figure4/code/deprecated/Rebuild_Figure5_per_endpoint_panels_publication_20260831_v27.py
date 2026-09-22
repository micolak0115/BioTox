#!/usr/bin/env python3
"""Figure 5 v27: aligned endpoints and refined axis typography.

This plotting-only revision preserves v26 while aligning all A-D endpoint
labels to the SR-ARE horizontal position, using uniform non-bold italic gene
labels, and enlarging the requested gene x-axis and GSEA y-axis text.
"""
from __future__ import annotations

from pathlib import Path

import Rebuild_Figure5_per_endpoint_panels_publication_20260831_v26 as publication


base = publication.base
base.OUT_DIR = Path(
    "/home/kyungan/scripts/BioTox/publication/final/figures_revised/Figure5/"
    "unadjusted_global_zbeta_exploratory_q010_publication_20260831_v27"
)
base.FIG_STEM = "Figure5_per_endpoint_panels_A4_q010_publication_20260831_v27"

# Use one fixed axes-relative x coordinate so different gene-name widths cannot
# shift the endpoint labels. This matches the SR-ARE position in v26.
base.ENDPOINT_LABEL_X = -0.465

# Gene names: identical styling across A-D, italic but no longer bold.
base.GENE_LABEL_FONTWEIGHT = "normal"
base.GENE_LABEL_FONTSTYLE = "italic"
base.GENE_LABEL_FONTSIZE = 11.3

# Requested axis-type enlargement while retaining the collision-free layout.
base.GENE_XLABEL_FONTSIZE = 12.2
base.GENE_TICK_FONTSIZE = 10.7
base.GSEA_YLABEL_FONTSIZE = 12.6
base.GSEA_TICK_FONTSIZE = 10.8


if __name__ == "__main__":
    base.main()
