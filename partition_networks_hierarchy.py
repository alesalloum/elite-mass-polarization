import os
import logging
import argparse

import numpy as np
import networkx as nx

from git.core_periphery_sbm import core_periphery as cp
from git.core_periphery_sbm import model_fit as mf
from config import (
    BEST_ASSORTATIVE_DIR,
    HIERARCHICAL_BATCH_HIERARCHY_DIR,
    N_SAMPLES,
    GIBBS_ITERATIONS,
    ensure_dirs,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_args():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Fit Core-Periphery SBM model to network data"
    )
    parser.add_argument(
        "network_name", help="Name of the network to process (format: topic_year)"
    )
    return parser.parse_args()


# Set random seed for reproducibility
np.random.seed(42)

# Helper functions


def load_graph(filename):
    """Load a graph from a GraphML file.

    Args:
        filename (str): Path to the GraphML file

    Returns:
        nx.Graph: Loaded graph

    Raises:
        Exception: If there is an error loading the graph
    """
    try:
        graph = nx.read_graphml(filename)
        logging.info(f"Loaded graph with {graph.number_of_nodes()} nodes")
        return graph
    except Exception as e:
        logging.error(f"Error loading graph: {e}")
        raise


# Model fitting functions


def fit_CP(graph):

    def get_hierarchical_groups(graph):

        # Get the 'polhierarchy-sbm' attribute for each node
        try:
            hierarchy_attr = nx.get_node_attributes(graph, "ppsbm")
        except KeyError:
            raise KeyError("The graph does not have the expected 'ppsbm' attribute.")

        # Categorize nodes into groups
        groups = {"A": set(), "B": set()}
        for node, hierarchy in hierarchy_attr.items():
            if hierarchy in groups:
                groups[hierarchy].add(node)
            else:
                print(
                    f"Warning: Unrecognized hierarchy '{hierarchy}' for node '{node}'."
                )

        # Check if all nodes have been categorized
        assert len(graph) == sum(
            len(group) for group in groups.values()
        ), "Some nodes were not categorized."

        return groups

    def encode_fourblocks(graph, A_node2label, B_node2label):

        A_CORE = "A_CORE"
        A_PERIPHERY = "A_PERIPHERY"
        B_CORE = "B_CORE"
        B_PERIPHERY = "B_PERIPHERY"

        fourblock_ms = dict()

        for node in graph.nodes:

            if node in A_node2label:
                if A_node2label[node] == 1:
                    fourblock_ms[node] = A_PERIPHERY
                else:
                    fourblock_ms[node] = A_CORE

            elif node in B_node2label:
                if B_node2label[node] == 1:
                    fourblock_ms[node] = B_PERIPHERY
                else:
                    fourblock_ms[node] = B_CORE

            else:
                print("Error!")

        assert len(fourblock_ms) == len(graph), "Inconsistency in assigning node blocks"
        nx.set_node_attributes(graph, fourblock_ms, name="polhierarchy-sbm")

        return graph

    # BOTH GROUPS
    polarized_groups = get_hierarchical_groups(graph)

    A_subgraph = nx.induced_subgraph(graph, polarized_groups["A"]).copy()
    B_subgraph = nx.induced_subgraph(graph, polarized_groups["B"]).copy()

    # Initialize hub-and-spoke model and infer structure
    logging.info(f"Fitting CP for group A...")
    A_hubspoke = cp.HubSpokeCorePeriphery(
        n_gibbs=GIBBS_ITERATIONS, n_mcmc=20 * len(A_subgraph)
    )
    A_hubspoke.infer(A_subgraph)

    logging.info(f"Fitting CP for group B...")
    B_hubspoke = cp.HubSpokeCorePeriphery(
        n_gibbs=GIBBS_ITERATIONS, n_mcmc=20 * len(B_subgraph)
    )
    B_hubspoke.infer(B_subgraph)

    A_node2label = A_hubspoke.get_labels(last_n_samples=50)
    B_node2label = B_hubspoke.get_labels(last_n_samples=50)

    G = encode_fourblocks(graph, A_node2label, B_node2label)

    A_node2label = A_hubspoke.get_labels(
        last_n_samples=50, prob=False, return_dict=False
    )
    B_node2label = B_hubspoke.get_labels(
        last_n_samples=50, prob=False, return_dict=False
    )

    A_dl = mf.mdl_hubspoke(A_subgraph, A_node2label, n_samples=100000)
    B_dl = mf.mdl_hubspoke(B_subgraph, B_node2label, n_samples=100000)

    return G, A_dl, B_dl


def run_pipeline(network_name):
    """Run the Core-Periphery SBM fitting pipeline for a given network.

    Args:
        network_name (str): Name of the network to process (format: topic_year)
    """
    ensure_dirs()

    logging.info(f"Fitting Core-Periphery SBM models for {network_name}")

    # Setup paths
    input_dir = os.path.join(BEST_ASSORTATIVE_DIR, network_name)
    input_path = os.path.join(input_dir, f"{network_name}.graphml")
    output_dir = os.path.join(HIERARCHICAL_BATCH_HIERARCHY_DIR, network_name)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load best assortative partition
        G = load_graph(input_path)

        # Fit multiple CP-SBM models
        for sample_idx in range(N_SAMPLES):
            logging.info(f"Fitting sample {sample_idx + 1}/{N_SAMPLES}")

            try:
                # Fit Core-Periphery model
                hG, hA_dl, hB_dl = fit_CP(G)
                logging.info(
                    f"Fitted CP-SBM with description lengths: A={hA_dl}, B={hB_dl}"
                )

                # Save results
                hG.graph["A_dl"] = hA_dl
                hG.graph["B_dl"] = hB_dl

                output_path = os.path.join(
                    output_dir, f"{network_name}_{sample_idx}.graphml"
                )
                nx.write_graphml(hG, output_path)
                logging.info(f"Saved model to {output_path}")

            except Exception as e:
                logging.error(f"Error in sample {sample_idx}: {e}")
                continue

    except Exception as e:
        logging.error(f"Error processing network {network_name}: {e}")


def main():
    """Main execution function."""
    args = parse_args()
    run_pipeline(args.network_name)


if __name__ == "__main__":
    main()
