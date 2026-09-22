#!/usr/bin/env python3
"""Figure 5 v29: NR-prefix-aligned endpoints and subtitle-aligned letters.

This plotting-only revision preserves v28 geometry and statistical semantics.
Endpoint-label centers align with the rendered center of the ``NR-`` prefix in
panel E, panel letters align vertically with their subplot subtitles, and the
redundant bottom caption is omitted.
"""
from __future__ import annotations

from pathlib import Path

import Rebuild_Figure5_per_endpoint_panels_publication_20260831_v28 as publication


base = publication.base
base.OUT_DIR = Path(
    "/home/kyungan/scripts/BioTox/publication/final/figures_revised/Figure5/"
    "unadjusted_global_zbeta_exploratory_q010_publication_20260831_v29"
)
base.FIG_STEM = "Figure5_per_endpoint_panels_A4_q010_publication_20260831_v29"

# Measured from the renderer: aligns the endpoint center with the center of the
# ``NR-`` prefix, an 18.125-pixel rightward shift from v28.
base.ENDPOINT_LABEL_X = -0.395888

base.ALIGN_PANEL_LETTERS_TO_TITLES = True
base.SHOW_FOOTNOTE = False


if __name__ == "__main__":
    base.main()
