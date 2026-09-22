# Deprecated Figure 4 rebuild scripts

The versioned `Rebuild_Figure5_per_endpoint_panels_*.py` files are historical
rebuilds and are no longer supported entry points. They are retained under
this directory for provenance. Their rendering logic has
been merged into [`generate_figure.py`](generate_figure.py), which is the only
canonical Figure 4 generator and is driven by `config.json`.

Deprecated files:

- `deprecated/Rebuild_Figure5_per_endpoint_panels_exploratory_q010_20260831_v24.py`
- `deprecated/Rebuild_Figure5_per_endpoint_panels_publication_20260831_v26.py`
- `deprecated/Rebuild_Figure5_per_endpoint_panels_publication_20260831_v27.py`
- `deprecated/Rebuild_Figure5_per_endpoint_panels_publication_20260831_v28.py`
- `deprecated/Rebuild_Figure5_per_endpoint_panels_publication_20260831_v29.py`
- `deprecated/Rebuild_Figure5_per_endpoint_panels_publication_20260831_v30.py`
- `deprecated/Rebuild_Figure5_per_endpoint_panels_publication_20260831_v31.py`
- `deprecated/Rebuild_Figure5_per_endpoint_panels_publication_20260831_v32.py`

They remain available for provenance and reproducibility of historical output,
but new runs must use `generate_figure.py`.
