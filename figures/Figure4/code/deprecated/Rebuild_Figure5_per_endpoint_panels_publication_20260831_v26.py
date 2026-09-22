#!/usr/bin/env python3
"""Figure 5 v26: publication-grade typography and panel proportions.

This plotting-only revision preserves the v25 statistical/display contract and
uses a wider gene-coefficient panel, a narrower compound-GSEA panel, explicit
inter-panel whitespace, larger endpoint labels, and enlarged typography.
"""
from __future__ import annotations

from pathlib import Path

import Rebuild_Figure5_per_endpoint_panels_exploratory_q010_20260831_v24 as base


base.OUT_DIR = Path(
    "/home/kyungan/scripts/BioTox/publication/final/figures_revised/Figure5/"
    "unadjusted_global_zbeta_exploratory_q010_publication_20260831_v26"
)
base.FIG_STEM = "Figure5_per_endpoint_panels_A4_q010_publication_20260831_v26"

# Statistical/display semantics inherited from v25.
base.EXPLORATORY_MARKER = "\u271d"
base.DISPLAY_GENE_SET_NAMES = {"Pperoxisome": "Peroxisome"}

# Publication layout: elongated coefficient plots, compact GSEA plots, and a
# visibly empty gutter between them.
base.FIG_SIZE_IN = (11.69, 16.1)
base.ROW_WIDTH_RATIOS = [0.98, 1.72]
base.ROW_WSPACE = 0.18
base.GENE_BOX_ASPECT = None
base.OUTER_HSPACE = 1.02
base.OUTER_LEFT = 0.115
base.OUTER_RIGHT = 0.93
base.OUTER_TOP = 0.982
base.OUTER_BOTTOM = 0.15
base.LETTER_X = -0.035
base.ENDPOINT_LABELPAD = 31
base.PANEL_LETTER_FONTSIZE = 29

# Larger type throughout, with emphasis on panel subtitles and endpoint names.
base.GENE_XLABEL_FONTSIZE = 10.8
base.GENE_TICK_FONTSIZE = 9.4
base.GENE_LABEL_FONTSIZE = 11.3
base.ENDPOINT_FONTSIZE = 16.5
base.SUBTITLE_FONTSIZE = 13.0
base.GSEA_XLABEL_FONTSIZE = 9.5
base.GSEA_YLABEL_FONTSIZE = 10.8
base.GSEA_TICK_FONTSIZE = 9.2
base.SIGNIFICANCE_MARKER_FONTSIZE = 17
base.EXPLORATORY_MARKER_FONTSIZE = 18
base.HEATMAP_ENDPOINT_FONTSIZE = 13.0
base.HEATMAP_SELECTED_LABEL_FONTSIZE = 9.7
base.HEATMAP_LABEL_FONTSIZE = 8.9
base.HEATMAP_MARKER_FONTSIZE = 17
base.HEATMAP_TITLE_FONTSIZE = 14.0
base.COLORBAR_TICK_FONTSIZE = 9.5
base.COLORBAR_LABEL_FONTSIZE = 10.8
base.FOOTNOTE_FONTSIZE = 9.2


if __name__ == "__main__":
    base.main()
