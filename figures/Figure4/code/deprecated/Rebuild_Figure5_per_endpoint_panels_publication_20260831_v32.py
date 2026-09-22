#!/usr/bin/env python3
"""Figure 5 v32: SR-MMP Peroxisome pathway panel.

This plotting-only revision preserves v31 and replaces the SR-MMP
``TNF-alpha Signaling via NF-kB`` compound-GSEA panel with the audited source
pathway ``Pperoxisome``, displayed with the corrected label ``Peroxisome``.
"""
from __future__ import annotations

from pathlib import Path

import Rebuild_Figure5_per_endpoint_panels_publication_20260831_v31 as publication


base = publication.base
base.OUT_DIR = Path(
    "/home/kyungan/scripts/BioTox/publication/final/figures_revised/Figure5/"
    "unadjusted_global_zbeta_exploratory_q010_publication_20260831_v32"
)
base.FIG_STEM = "Figure5_per_endpoint_panels_A4_q010_publication_20260831_v32"
base.PANEL_PATHWAY = {
    **base.PANEL_PATHWAY,
    "SR-MMP": "Pperoxisome",
}


if __name__ == "__main__":
    base.main()
