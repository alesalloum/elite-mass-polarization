"""
Global configuration and constants for the data pipeline.
"""

import os

# Data paths
DATA_DIR = "./data"
NETWORKS_DIR = "./networks"
RESULTS_DIR = "./results"

# Network directories
NETWORKS_MULTIDIRECTED_DIR = os.path.join(NETWORKS_DIR, "networks_multidirected")
OVERLAPS_DIR = os.path.join(NETWORKS_MULTIDIRECTED_DIR, "overlaps")
NO_OVERLAPS_DIR = os.path.join(NETWORKS_MULTIDIRECTED_DIR, "no_overlaps")
HIERARCHICAL_BATCH_ASSORTATIVE_DIR = os.path.join(
    NETWORKS_DIR, "hierarchical_batch_assortative"
)
HIERARCHICAL_BATCH_HIERARCHY_DIR = os.path.join(
    NETWORKS_DIR, "hierarchical_batch_hierarchy"
)
BEST_ASSORTATIVE_DIR = os.path.join(NETWORKS_DIR, "best_assortative")
BEST_HIERARCHY_DIR = os.path.join(NETWORKS_DIR, "best_hierarchy")

# Topics and years
TOPICS = ["climate", "immigration", "social", "economy", "education"]
YEARS = ["19", "23"]  # 2019 and 2023

# Time periods
ELECTION_PERIODS = {
    "19": {"start": "2019-01-21 00:00:00", "end": "2019-04-15 00:00:00"},
    "23": {"start": "2023-01-09 00:00:00", "end": "2023-04-03 00:00:00"},
}

# Model parameters
N_SAMPLES = 50  # Number of samples for model fitting
MCMC_ITERATIONS = 100  # Number of MCMC iterations
GIBBS_ITERATIONS = 200  # Number of Gibbs iterations

# Network analysis parameters
BOOTSTRAP_ITERATIONS = 500  # Number of bootstrap iterations for alignment analysis

# Null model parameters
NULL_MODEL_SAMPLES = 5  # Number of samples for null model generation
NULL_MODEL_METHODS = {
    "zerok": "random_graph",  # Erdős-Rényi random graph
    "onek": "configuration_model",  # Configuration model preserving degree sequence
}

# METIS partitioning parameters
METIS_OPTIONS = {
    "ufactor": 400,  # Maximum load imbalance
    "niter": 100,  # Number of refinement iterations
    "contig": True,  # Force contiguous partitions
}


def ensure_dirs():
    """Create all necessary directories if they don't exist."""
    dirs = [
        DATA_DIR,
        NETWORKS_DIR,
        RESULTS_DIR,
        NETWORKS_MULTIDIRECTED_DIR,
        OVERLAPS_DIR,
        NO_OVERLAPS_DIR,
        HIERARCHICAL_BATCH_ASSORTATIVE_DIR,
        HIERARCHICAL_BATCH_HIERARCHY_DIR,
        BEST_ASSORTATIVE_DIR,
        BEST_HIERARCHY_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
