import os
import json
import logging
import argparse

import numpy as np
import pymetis
import networkx as nx

import polarization_algorithms as pol
from config import (
    NO_OVERLAPS_DIR,
    RESULTS_DIR,
    ensure_dirs,
    NULL_MODEL_SAMPLES,
    NULL_MODEL_METHODS,
    METIS_OPTIONS,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_args():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Compute interactional null model for network polarization"
    )
    parser.add_argument(
        "network_name", help="Name of the network to analyze (format: topic_year)"
    )
    return parser.parse_args()


# Helper functions


def load_graph(filename):
    """Load a graph from a GraphML file.

    Args:
        filename (str): Path to the GraphML file

    Returns:
        nx.Graph: Loaded network graph

    Raises:
        Exception: If there is an error loading the graph
    """
    try:
        G = nx.read_graphml(filename)
        logging.info(
            f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
        )
        return G
    except Exception as e:
        logging.error(f"Error loading graph: {e}")
        raise


def prepare_network(graph):
    """Prepare network for analysis by preprocessing and standardizing.

    Preprocessing steps:
    1. Convert to undirected
    2. Remove self-loops
    3. Extract giant component
    4. Convert node labels to integers

    Args:
        graph (nx.Graph): Input network

    Returns:
        nx.Graph: Preprocessed network
    """
    # Convert to undirected and remove self-loops
    graph = graph.to_undirected()
    graph.remove_edges_from(nx.selfloop_edges(graph))

    # Extract giant component
    largest_cc = max(nx.connected_components(graph), key=len)
    GC = graph.subgraph(largest_cc).copy()

    # Standardize node labels
    GC = nx.convert_node_labels_to_integers(
        GC, first_label=0, ordering="default", label_attribute="user_id"
    )

    logging.info(
        f"Prepared network: {GC.number_of_nodes()} nodes, "
        f"{GC.number_of_edges()} edges"
    )
    return GC


def get_adjacency_dict(graph):
    """Convert graph to adjacency list format.

    Args:
        graph (nx.Graph): Input network

    Returns:
        dict: Adjacency list representation
    """
    return {node: list(graph.neighbors(node)) for node in graph.nodes}


def partition_metis(graph):
    """Partition graph using METIS algorithm.

    Args:
        graph (nx.Graph): Input network

    Returns:
        tuple: (number of cuts, node membership dictionary)
    """
    # Convert to adjacency list format
    adj_dict = get_adjacency_dict(graph)
    adj_list = [np.asarray(neighs) for neighs in adj_dict.values()]

    # Run METIS partitioning
    n_cuts, membership = pymetis.part_graph(
        nparts=2,
        adjacency=adj_list,
        options=pymetis.Options(**METIS_OPTIONS),
    )  # Convert to node-based dictionary
    membership = dict(zip(adj_dict.keys(), membership))
    logging.info(f"METIS partitioning: {n_cuts} cuts")

    return n_cuts, membership


def compute_polarization(graph, membership):
    """Compute polarization metrics for a network partition.

    Args:
        graph (nx.Graph): Input network
        membership (dict): Node membership dictionary

    Returns:
        float: AEI polarization score
    """
    AEI = pol.AEI_polarization(graph, ms=membership)
    logging.info(f"Computed polarization: AEI = {AEI}")
    return AEI


def shuffled_polarization(graph, n_samples=NULL_MODEL_SAMPLES, method="zerok"):
    """Generate null models and compute their polarization scores.

    Args:
        graph (nx.Graph): Input network
        n_samples (int): Number of null models to generate
        method (str): Null model type ('zerok' or 'onek')

    Returns:
        tuple: (mean polarization, std polarization)

    Raises:
        ValueError: If invalid method specified
    """
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    buffer = []

    if method not in NULL_MODEL_METHODS:
        raise ValueError(
            f"Invalid method. Supported methods: {list(NULL_MODEL_METHODS.keys())}"
        )

    for sample_idx in range(n_samples):
        logging.info(f"Generating null model {sample_idx + 1}/{n_samples} ({method})")

        if method == "zerok":
            R = nx.gnm_random_graph(n, m)
        elif method == "onek":
            degree_sequence = [d for _, d in graph.degree()]
            R = nx.Graph(nx.configuration_model(deg_sequence=degree_sequence))
            R.remove_edges_from(nx.selfloop_edges(R))

        R = prepare_network(R)
        _, membership = partition_metis(R)
        polarization_scores = compute_polarization(R, membership)
        buffer.append(polarization_scores)

    polarization_scores_mean = np.mean(buffer, axis=0)
    polarization_scores_std = np.std(buffer, axis=0)

    return polarization_scores_mean, polarization_scores_std


def save_results(results, network_name):
    """Save polarization results to a JSON file.

    Args:
        results (dict): Dictionary of polarization metrics
        network_name (str): Name of the analyzed network
    """
    output_dir = os.path.join(RESULTS_DIR, "interactional")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{network_name}.json")
    with open(output_path, "w") as json_file:
        json.dump(results, json_file, indent=4)
    logging.info(f"Saved results to {output_path}")


def run_pipeline(network_name):
    """Run the interactional null model analysis pipeline.

    Args:
        network_name (str): Name of the network to analyze
    """
    # Setup paths
    input_path = os.path.join(NO_OVERLAPS_DIR, f"{network_name}.graphml")

    try:
        # Load and prepare network
        logging.info(f"Processing network: {network_name}")
        G = load_graph(input_path)
        G = prepare_network(G)

        # Generate partitions
        _, membership = partition_metis(G)

        # Compute observed polarization
        AEI_observed = compute_polarization(G, membership=membership)

        # Generate null models
        AEI_mean_onek, AEI_std_onek = shuffled_polarization(
            G, n_samples=5, method="onek"
        )

        # Prepare results
        results = {
            "observed": float(AEI_observed),
            "onek_average": float(AEI_mean_onek),
            "onek_std": float(AEI_std_onek),
            "normalized": float(AEI_observed - AEI_mean_onek),
        }

        # Save results
        save_results(results, network_name)
        logging.info("Analysis completed successfully")

    except Exception as e:
        logging.error(f"Error processing {network_name}: {e}")
        raise


def main():
    """Main execution function."""
    args = parse_args()
    ensure_dirs()
    run_pipeline(args.network_name)


if __name__ == "__main__":
    main()
