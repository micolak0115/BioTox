# LaTeX recipe for Table S8

`table_s8_figure3_panel_statistics.tex` is a landscape `longtable` containing
all 48 endpoint-cell contexts. Add these packages to the manuscript preamble:

```latex
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{pdflscape}
```

Insert the table where supplementary Table S8 should appear:

```latex
\input{table_s8_figure3_panel_statistics.tex}
```

The TeX source already includes the `landscape`, `scriptsize`, spacing,
caption, label, repeated header, NR/SR block separation, and table rules.
Use `table_s8_figure3_panel_statistics_display.csv` for journal production
systems that ingest tables separately from the manuscript. The numeric CSV
retains unrounded values and is the source of record.
