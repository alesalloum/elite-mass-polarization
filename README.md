# Elite-Mass Polarization

A reproducible data pipeline for building and analyzing topic-specific retweet networks and measuring political polarization with hierarchies.

This repository contains scripts to:
- generate topic-specific retweet networks from Parquet tweet data,
- fit partition models (PP-SBM) to identify polarized groups,
- fit core-periphery (CP-SBM) models inside groups to uncover hierarchical roles,
- compute alignment between topic networks,
- compute interactional null models and polarization metrics.

This README summarizes the project layout, key workflows, configuration, and example commands.

## Big picture

1. Input: a Parquet file of retweet edges with topic flags and timestamps (`./data/relevant_retweets.parquet`).
2. Network generation: topic-specific MultiDiGraph retweet networks (overlapping and strict no-overlap variants) are created and stored as GraphML.
3. Partitioning: two-stage pipeline:
   - PP-SBM (assortative) partitioning to split users into A/B groups.
   - Core-periphery (hub-and-spoke) model inside each group to identify core/periphery roles.
4. Analysis: compute alignment, compute interactional null models (degree-preserving nulls), and compute polarization metrics (AEI, EI, RWC, etc.).
5. Output: GraphML networks and JSON results stored under `./networks*/` and `./results/`.

## Repository layout (key files)

- `generate_networks.py` — create topic networks from `./data/relevant_retweets.parquet`.
- `partition_networks_group.py` — fit PP-SBM models (assortative) and save multiple samples.
- `select_best_assortative.py` — choose the best PP-SBM sample by description length.
- `partition_networks_hierarchy.py` — fit core-periphery SBM inside A/B groups using best assortative network.
- `compute_alignment.py` — bootstrap alignment (NMI, JI) between two networks.
- `compute_interactional_null.py` — generate null models (configuration / random graphs) and compute AEI.
- `polarization_algorithms.py` — implementations of polarization metrics used across scripts.
- `config.py` — centralized configuration: paths, topics, years, model parameters.
- `visualize_decomposition.py` — create AEI decomposition and group-size visualizations across topics; reads hierarchical GraphMLs from `./networks/hierarchical/` and draws the decomposition/group-size figure.

## Naming conventions

- Networks are named `{topic}_{year}` where `topic` is one of `climate, immigration, social, economy, education`, and `year` is `19` or `23`.
  - Example: `climate_19`.
- Graph files are GraphML: `./networks_multidirected/overlaps/climate_19.graphml` and `./networks_multidirected/no_overlaps/climate_19.graphml`.

## Centralized configuration

`config.py` exposes constants used by all scripts. Important keys:
- DATA_DIR: `./data`
- NETWORKS_DIR: `./networks`
- OVERLAPS_DIR / NO_OVERLAPS_DIR: where generated GraphMLs are written
- HIERARCHICAL_BATCH_ASSORTATIVE_DIR: where PP-SBM samples are saved
- BEST_ASSORTATIVE_DIR: where the selected best PP-SBM network is saved
- HIERARCHICAL_BATCH_HIERARCHY_DIR: where CP model samples are saved
- RESULTS_DIR: `./results`
- TOPICS, YEARS, ELECTION_PERIODS
- N_SAMPLES, MCMC_ITERATIONS, GIBBS_ITERATIONS (for partitions)
- BOOTSTRAP_ITERATIONS (alignment)
- NULL_MODEL_SAMPLES, NULL_MODEL_METHODS (interactional nulls)
- METIS_OPTIONS (used by METIS partitioning via pymetis)

All scripts import values from `config.py` and use `ensure_dirs()` to create required folders.

## Scripts: inputs / outputs and parameters

- `generate_networks.py` reads `./data/relevant_retweets.parquet` and writes GraphMLs to `OVERLAPS_DIR` and `NO_OVERLAPS_DIR`.
- `partition_networks_group.py` reads a no-overlap GraphML from `NO_OVERLAPS_DIR` and writes many samples to `HIERARCHICAL_BATCH_ASSORTATIVE_DIR`.
- `select_best_assortative.py` reads samples in `HIERARCHICAL_BATCH_ASSORTATIVE_DIR` and writes selected best to `BEST_ASSORTATIVE_DIR`.
- `partition_networks_hierarchy.py` reads the best assortative GraphML from `BEST_ASSORTATIVE_DIR` and writes CP samples to `HIERARCHICAL_BATCH_HIERARCHY_DIR`.
- `compute_alignment.py` loads hierarchical GraphMLs (the scripts expect `polhierarchy-sbm` node attributes) and outputs JSON under `./results/alignment/`.
- `compute_interactional_null.py` loads no-overlap GraphML, prepares giant component, partitions with METIS, computes AEI, and saves JSON under `./results/interactional/`.

Parameters such as number of samples, number of bootstrap iterations, and null-model sample size are defined in `config.py` so you can tune them centrally.

## Polarization metrics

Available in `polarization_algorithms.py` (used by the analysis scripts):
- RWC (Random Walk Controversy)
- EI (E-I Index)
- AEI (Adaptive E-I Index) — used by `compute_interactional_null.py`
- BCC (Betweenness Centrality Controversy)
- BP (Boundary Polarization / GMCK)
- DP (Dipole Polarization / MBLB)
- Q (Modularity)

## Notes and Caveats

- Many scripts assume GraphML files contain specific node attributes after model fitting (e.g., `ppsbm`, `polhierarchy-sbm`). Keep the pipeline order: generate -> PP-SBM -> select best -> CP-SBM.
- For reproducibility, some randomness is seeded when scripts call numpy's RNG; however full determinism may require seeding in all modules and external libraries.
- [The prerocessed network data used in the paper can be found here](https://zenodo.org/records/17446451)

## Troubleshooting

- If a script fails to find a GraphML, check the expected directory in `config.py` and confirm the file was written.
- If `graph_tool` import fails, try installing via conda: `conda install -c conda-forge graph-tool`.
- For `pymetis`, ensure libmetis development headers are present or install via pip on platforms that support binary wheels.
