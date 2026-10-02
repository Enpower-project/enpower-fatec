# FATEC Architecture Diagram

![FATEC architecture diagram](architecture.svg)

Rendered diagram: [`architecture.svg`](architecture.svg) (committed) / [`architecture.png`](architecture.png) (local preview, git-ignored).
Regenerate either one with:

```sh
python docs/diagrams/render_architecture.py
```

## Component description

The service exposes REST endpoints that persist tasks in PostgreSQL and run the heavy computation asynchronously via a FastAPI background task.

- **Step 1 — Elasticity (`POST /execute`):** `run_consumers` (in `main_elasticity.py`) preprocesses partner data (`processing/DataPreprocessor.py`) enriched with Open-Meteo weather, then `ElasticityModel` (`causality/ElasticityAnalysis.py`, ECONML DML + XGBoost) estimates each consumer's 24h price elasticity of demand, a flexibility ranking and p-values.
- **Step 2 — Clustering (`POST /clustering`):** `run_clustering` (in `main_clusters.py`) fetches the completed elasticity task over HTTP (`API_BASE_URL`), merges consumption, elasticity and synthetic EWH profiles, and selects the number of clusters with KMeans + silhouette scoring.
- **Step 3 — EWH optimization (`POST /ewh_optimization`):** `run_ewh_optimization` (in `main_ewh.py`) fetches the completed clustering task over HTTP and optimizes the electric water heater operating calendar for each cluster mean profile (MILP via PuLP/CBC, `ewh/ewh_flex`).

All steps write their artifacts (input JSON, log, plots, result JSON) to a timestamped folder under `results/`, and their status/results JSON to the matching PostgreSQL table (`tasks`, `clustering_tasks`, `ewh_tasks`).
