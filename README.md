# Operation Greens and Mandi Price Volatility — Replication Package

Replication materials for:

> Dagar, H. *Did Operation Greens Stabilize Food Prices? Causal Evidence from India's Mandi Markets.* Submitted to *Agricultural Economics*.

The package reproduces every table and figure in the paper. Raw price data are public and are not redistributed here (see `01-data/README.md`).

## Layout

The folder mirrors the original project, so the scripts run unmodified (each script resolves paths relative to the package root).

| Path | Contents |
|------|----------|
| `01-data/` | Empty on download. Place the raw Agmarknet parquet files and GADM boundaries here (instructions inside). |
| `02-analysis/*.py` | Analysis pipeline (run order below). |
| `02-analysis/*.csv, *.parquet, *.json` | Derived outputs exactly as used in the paper. |
| `03-drafts/figures/` | The ten figures in the paper (PNG, 300 dpi). |
| `MANIFEST.sha256` | SHA-256 hashes of the shipped derived files and of the raw 2013–2024 input files used. |
| `run_all.py` | One-command reproduction. |

## Pipeline

| Step | Script | Needs | Produces | Paper output |
|------|--------|-------|----------|--------------|
| 1 | `volatility_analysis.py` | raw parquet | `cv_panel.parquet`, `volatility_*.csv`, `summary_stats.csv` | Tables 1–2, Figs 1, 3–5, 7 inputs |
| 2 | `temporal_spatial_analysis.py` | step 1, GADM | `stl_components.csv`, `structural_breaks.csv`, `moran_global.csv`, `moran_local.csv` | Table 3, Table C1, Figs 2, 6, 9 inputs |
| 3 | `dml_operation_greens.py` | step 1, raw parquet | `dml_results_og.json`, `dml_overlap.csv`, `cate_og.parquet`, `dml_cate_district.csv` | Table 4, §6.1, §6.4 |
| 4 | `table4_clustered_se.py` | step 1, raw parquet | `table4_clustered_se.json` | Table 4 mandi-clustered 95% CIs; Figure 8 CIs |
| 5 | `revision_analysis.py` | step 1, raw parquet, GADM | `revision_results.json`, `fig10_event_study.png` | Event study, balanced panel, pre-expansion window, MDE, residual Moran's I, Table A2 |
| 6 | `visualizations.py` | steps 1–3 | Figs 4, 6, 7 | Figures 4, 6, 7 |
| 7 | `figures_v2.py` | steps 1–4, GADM | Figs 1, 2, 3, 5, 8, 9 | Figures 1, 2, 3, 5, 8, 9 |

`visualizations.py` also writes earlier versions of some figures that the paper does not use; the paper uses the `fig*_v2_*` files where both exist.

Table 4's confidence intervals are the mandi-clustered 95% CIs in `table4_clustered_se.json`; `dml_results_og.json` also stores the unclustered (i.i.d.) SEs, which are roughly 2.5-3x smaller. Figure 8 is drawn from the clustered CIs.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt

python run_all.py               # full pipeline (needs raw data in 01-data/)
python run_all.py --figures-only  # rebuild figures from the shipped derived files
```

Original run: Python 3.13, Windows 11. Learners use `random_state=0`; NumPy is seeded with 42. Steps 3–5 are the slow ones (cross-fitted LightGBM, Random Forest, and causal forest on about 250,000 mandi-crop-months); step 4 adds about three minutes for the clustered-SE reruns.

## Data sources

- **Prices:** Agmarknet daily arrivals and modal prices (Government of India), via the Kaggle archive "Daily Market Prices of Commodity India (2001–2026)", https://www.kaggle.com/datasets/khandelwalmanas/daily-commodity-prices-india, Government Open Data License – India. The archive is updated over time; `MANIFEST.sha256` lists the hashes of the 2013–2024 files used.
- **Boundaries:** GADM v4.1, India level 2 (`gadm41_IND_2.json`), https://gadm.org. GADM's licence does not allow redistribution, so download it directly.

## Citation

Cite the paper and this archive: `[Zenodo DOI — added on first release]`. See `CITATION.cff`.

## Licence

Code: MIT (`LICENSE`). Derived data: CC-BY-4.0; the original source terms of Agmarknet and GADM still apply.

## Contact

Harsh Dagar — harshdagar5879@gmail.com — ORCID [0009-0008-7394-130X](https://orcid.org/0009-0008-7394-130X)
ICAR–National Dairy Research Institute, Karnal, India
