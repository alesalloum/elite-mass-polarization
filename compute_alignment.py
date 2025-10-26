import argparse
import json
import logging
import random
import os

import networkx as nx
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score

from config import (
    RESULTS_DIR,
    ensure_dirs,
    BOOTSTRAP_ITERATIONS,
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
        description="Compute alignment between two topic networks"
    )
    parser.add_argument(
        "network_name_1", help="Name of first network (format: topic_year)"
    )
    parser.add_argument(
        "network_name_2", help="Name of second network (format: topic_year)"
    )
    parser.add_argument("year", help="Year for comparison (19 or 23)")
    return parser.parse_args()


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
        logging.info(f"Loaded graph with {G.number_of_nodes()} nodes")
        return G
    except Exception as e:
        logging.error(f"Error loading graph: {e}")
        raise


def get_info(G):

    ms = nx.get_node_attributes(G, "polhierarchy-sbm")

    A_core = {key for key in ms if ms[key] == "A_CORE"}
    A_periphery = {key for key in ms if ms[key] == "A_PERIPHERY"}

    B_core = {key for key in ms if ms[key] == "B_CORE"}
    B_periphery = {key for key in ms if ms[key] == "B_PERIPHERY"}

    assert len(G) == len(A_core) + len(B_core) + len(A_periphery) + len(B_periphery)

    return A_core, A_periphery, B_core, B_periphery, len(G)


def get_network_structure(G):
    """Extract core-periphery structure from a network.

    Args:
        G (nx.Graph): Network graph with polhierarchy-sbm attributes

    Returns:
        tuple: Sets of nodes for each structural component and total size
            (A_core, A_periphery, B_core, B_periphery, N)

    Raises:
        KeyError: If polhierarchy-sbm attributes are missing
        AssertionError: If node counts don't match
    """
    try:
        ms = nx.get_node_attributes(G, "polhierarchy-sbm")

        A_core = {key for key in ms if ms[key] == "A_CORE"}
        A_periphery = {key for key in ms if ms[key] == "A_PERIPHERY"}
        B_core = {key for key in ms if ms[key] == "B_CORE"}
        B_periphery = {key for key in ms if ms[key] == "B_PERIPHERY"}

        total_nodes = len(G)
        component_sum = len(A_core) + len(B_core) + len(A_periphery) + len(B_periphery)

        assert total_nodes == component_sum, (
            f"Node count mismatch: graph has {total_nodes} nodes but "
            f"components sum to {component_sum}"
        )

        return A_core, A_periphery, B_core, B_periphery, total_nodes

    except KeyError:
        logging.error("Missing polhierarchy-sbm attributes in graph")
        raise


def bootstrap_network(G):

    A_core, A_periphery, B_core, B_periphery, N = get_info(G)

    A_core = list(A_core)
    A_periphery = list(A_periphery)
    B_core = list(B_core)
    B_periphery = list(B_periphery)

    p_A_core = len(A_core) / N
    p_B_core = len(B_core) / N

    p_A_periphery = len(A_periphery) / N
    p_B_periphery = len(B_periphery) / N

    bootstrapped_hierarchies = np.random.choice(
        ["A_core", "B_core", "A_periphery", "B_periphery"],
        N,
        p=[p_A_core, p_B_core, p_A_periphery, p_B_periphery],
    )

    A_core_bootstrap, A_periphery_bootstrap, B_core_bootstrap, B_periphery_bootstrap = (
        [],
        [],
        [],
        [],
    )

    for hier_class in bootstrapped_hierarchies:

        if hier_class == "A_core":
            random_node = random.sample(A_core, k=1)[0]
            A_core_bootstrap.append(random_node)

        elif hier_class == "B_core":
            random_node = random.sample(B_core, k=1)[0]
            B_core_bootstrap.append(random_node)

        elif hier_class == "A_periphery":
            random_node = random.sample(A_periphery, k=1)[0]
            A_periphery_bootstrap.append(random_node)

        elif hier_class == "B_periphery":
            random_node = random.sample(B_periphery, k=1)[0]
            B_periphery_bootstrap.append(random_node)

        else:
            print("Error")

    return (
        A_core_bootstrap,
        A_periphery_bootstrap,
        B_core_bootstrap,
        B_periphery_bootstrap,
    )


def compute_alignment_for_pair(net1, net2):

    network_dir_1 = f"./networks/hierarchical/{net1}.graphml"
    network_dir_2 = f"./networks/hierarchical/{net2}.graphml"

    G_observed_1 = load_graph(network_dir_1)
    G_observed_2 = load_graph(network_dir_2)

    bootstrap_results = []

    for i in range(BOOTSTRAP_ITERATIONS):

        if i % (BOOTSTRAP_ITERATIONS // 5) == 0:
            logging.info(f"At iteration {i}/{BOOTSTRAP_ITERATIONS}")

        # Bootstrap operations
        (
            A_core_bootstrap_1,
            A_periphery_bootstrap_1,
            B_core_bootstrap_1,
            B_periphery_bootstrap_1,
        ) = bootstrap_network(G_observed_1)
        (
            A_core_bootstrap_2,
            A_periphery_bootstrap_2,
            B_core_bootstrap_2,
            B_periphery_bootstrap_2,
        ) = bootstrap_network(G_observed_2)

        A_core_1 = set(A_core_bootstrap_1)
        A_periphery_1 = set(A_periphery_bootstrap_1)
        B_core_1 = set(B_core_bootstrap_1)
        B_periphery_1 = set(B_periphery_bootstrap_1)

        A_core_2 = set(A_core_bootstrap_2)
        A_periphery_2 = set(A_periphery_bootstrap_2)
        B_core_2 = set(B_core_bootstrap_2)
        B_periphery_2 = set(B_periphery_bootstrap_2)

        # Alignment computations
        common_core = (A_core_1 | B_core_1) & (A_core_2 | B_core_2)
        common_periphery = (A_periphery_1 | B_periphery_1) & (
            A_periphery_2 | B_periphery_2
        )

        core_stances_1 = dict()
        core_stances_2 = dict()

        for corenode in common_core:

            if corenode in A_core_1:
                core_stances_1[corenode] = "A"
            elif corenode in B_core_1:
                core_stances_1[corenode] = "B"
            else:
                print("Error")
                break

            if corenode in A_core_2:
                core_stances_2[corenode] = "A"
            elif corenode in B_core_2:
                core_stances_2[corenode] = "B"
            else:
                print("Error")
                break

        periphery_stances_1 = dict()
        periphery_stances_2 = dict()

        for peripherynode in common_periphery:

            if peripherynode in A_periphery_1:
                periphery_stances_1[peripherynode] = "A"
            elif peripherynode in B_periphery_1:
                periphery_stances_1[peripherynode] = "B"
            else:
                print("Error")
                break

            if peripherynode in A_periphery_2:
                periphery_stances_2[peripherynode] = "A"
            elif peripherynode in B_periphery_2:
                periphery_stances_2[peripherynode] = "B"
            else:
                print("Error")
                break

        core_stances_1_list = list(core_stances_1.values())
        core_stances_2_list = list(core_stances_2.values())

        NMI_core = normalized_mutual_info_score(
            core_stances_1_list, core_stances_2_list
        )
        JI_core = len(common_core) / len((A_core_1 | B_core_1) | (A_core_2 | B_core_2))

        periphery_stances_1_list = list(periphery_stances_1.values())
        periphery_stances_2_list = list(periphery_stances_2.values())

        NMI_periphery = normalized_mutual_info_score(
            periphery_stances_1_list, periphery_stances_2_list
        )
        JI_periphery = len(common_periphery) / len(
            (A_periphery_1 | B_periphery_1) | (A_periphery_2 | B_periphery_2)
        )

        bootstrap_results.append([NMI_periphery, NMI_core, JI_periphery, JI_core])

    results_table = np.asarray(bootstrap_results)

    return results_table


def save_results(results_table, network1, network2, year):
    """Save alignment results to a JSON file.

    Args:
        results_table (np.ndarray): Table of bootstrap results
        network1 (str): Name of first network
        network2 (str): Name of second network
        year (str): Year of comparison
    """
    # Calculate summary statistics
    alignment_results = {
        "NMI_periphery_ave": float(np.mean(results_table[:, 0])),
        "NMI_periphery_std": float(np.std(results_table[:, 0])),
        "NMI_core_ave": float(np.mean(results_table[:, 1])),
        "NMI_core_std": float(np.std(results_table[:, 1])),
        "JI_periphery_ave": float(np.mean(results_table[:, 2])),
        "JI_periphery_std": float(np.std(results_table[:, 2])),
        "JI_core_ave": float(np.mean(results_table[:, 3])),
        "JI_core_std": float(np.std(results_table[:, 3])),
    }

    # Setup output directory
    output_dir = os.path.join(RESULTS_DIR, "alignment")
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    output_path = os.path.join(
        output_dir, f"{network1[:-3]}-{network2[:-3]}_{year}_alignment.json"
    )

    with open(output_path, "w") as fp:
        json.dump(alignment_results, fp, indent=2)
    logging.info(f"Saved alignment results to {output_path}")


def main():
    """Main execution function."""
    # Parse arguments and ensure directories exist
    args = parse_args()
    ensure_dirs()

    logging.info(
        f"Processing alignment between {args.network_name_1} and {args.network_name_2}"
    )

    try:
        # Compute alignment
        results = compute_alignment_for_pair(args.network_name_1, args.network_name_2)

        # Save results
        save_results(results, args.network_name_1, args.network_name_2, args.year)
        logging.info("Alignment analysis completed successfully")

    except Exception as e:
        logging.error(f"Error in alignment computation: {e}")
        raise


if __name__ == "__main__":
    main()
