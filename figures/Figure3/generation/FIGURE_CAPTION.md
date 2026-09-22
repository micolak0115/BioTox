# Figure 3 caption

**Figure 3 | Cellular and temporal context dependence of transcriptomic
information.** (A) NR-Aromatase. (B) SR-ARE. (C) SR-MMP.
(D) SR-p53. Within each assay panel, results
are grouped first by exposure duration and then by biological context. The
transcriptomic input is the unadjusted (`standardized`) landmark-gene
representation. The
6 h group contains all four primary full-cohort cell lines (HA1E, HepG2,
HT29 and MCF7), whereas the 24 h group contains the three eligible
paired-timepoint cell lines (HA1E, HT29 and MCF7). HepG2 was unavailable at
24 h because its paired cohort was excluded before endpoint modelling. A
solid vertical rule separates the exposure groups.

The left subplot in each panel reports mean outer-test area under
the precision-recall curve (AUPRC) for the shared-intercept transcriptomic-term-off
molecular comparator (left, hatched bar) and the molecular-plus-transcriptomic model
(right, solid bar). Error bars show the sample standard deviation
across 20 repeat-level means, each averaged over five outer folds. Prominent
arrows connect paired bar heights, and labels report the signed AUPRC change.
Exactly one cell-line pair within each assay-by-exposure group, selected by
the highest absolute AUPRC across both models and all eligible cell lines, is
highlighted in firebrick for both the molecular and molecular-plus-
transcriptomic bars. Its paired gain label is bold. All other bars are grey.

The open rectangular tick-label box identifies the cell line used by the
corresponding Tox21 reporter assay. No cell-line colour coding is used.

The right subplot reports mean paired improvement over the
recalibrated frozen chemical predictor (Delta AUPRC). Error bars show
Nadeau-Bengio-corrected 95% confidence intervals and the dotted horizontal
line denotes no incremental gain. Positive gains with BH q<0.05 are
confirmatory and firebrick; positive gains with 0.05<=q<0.10 are exploratory
and ochre; other contexts are grey. Stars indicate confirmatory BH q-value
tiers (*q <= 0.05, **q <= 0.01 and ***q <= 0.001); a dagger marks
exploratory 0.05<=q<0.10.

Separate legends are centered beneath the two subplot columns. The left
legend identifies the chemical and chemical-plus-transcriptomic bars and the
best-performance highlighting. The right legend identifies non-significant
(q>=0.10), exploratory (0.05<=q<0.10), and confirmatory (q<0.05) incremental
gain categories.

Coloring is descriptive and was not used for model fitting, hyperparameter
selection or statistical inference. Error bars in the absolute and
incremental subplots quantify different quantities and should not be
compared directly. BH q-values are calculated within each analysis cohort;
nominal P values remain in the source table but do not determine figure
significance. Because the full 6 h and paired 24 h
cohorts differ, visual comparisons between exposure groups are descriptive
and are not paired timepoint estimates. AUPRC, area under the
precision-recall curve.
