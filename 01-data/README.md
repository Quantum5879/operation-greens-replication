# Raw data (not redistributed)

Place two inputs here before running the full pipeline.

1. **Agmarknet price panel.** Download the Kaggle dataset
   https://www.kaggle.com/datasets/khandelwalmanas/daily-commodity-prices-india
   and extract it so the yearly files sit at

       01-data/archive (4)/parquet/2013.parquet ... 2024.parquet

   (the folder name `archive (4)` matches the original project; keep it so the scripts run unmodified).
   Only 2013–2024 are read. Compare your files with the hashes in `../MANIFEST.sha256`; the Kaggle
   archive is updated over time, so later downloads may differ slightly.

2. **GADM v4.1 India level 2.** Download `gadm41_IND_2.json` from https://gadm.org/download_country.html
   and place it at

       01-data/geo/gadm41_IND_2.json
