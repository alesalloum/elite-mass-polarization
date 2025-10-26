import os
import logging
import argparse

import networkx as nx
import graph_tool.all as gt

from config import (
    NO_OVERLAPS_DIR,
    HIERARCHICAL_BATCH_ASSORTATIVE_DIR,
    N_SAMPLES,
    MCMC_ITERATIONS,
    ensure_dirs,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fit PP-SBM models to network data")
    parser.add_argument(
        "network_name", help="Name of the network to process (format: topic_year)"
    )
    return parser.parse_args()


# Helper functions


def load_graph(filename):
    """Load and preprocess a graph from a GraphML file.

    Args:
        filename (str): Path to the GraphML file

    Returns:
        nx.Graph: Preprocessed undirected graph (largest connected component)
    """
    try:
        # Read and preprocess the graph
        G_directed = nx.read_graphml(filename)
        G_undirected = G_directed.to_undirected()
        G_undirected.remove_edges_from(nx.selfloop_edges(G_undirected))

        # Extract largest connected component
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        G = G_undirected.subgraph(largest_cc).copy()

        logging.info(
            f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
        )
        return G

    except Exception as e:
        logging.error(f"Error loading graph: {e}")
        raise


def nx_2_gt(G_nx):

    # Create a graph-tool graph from the NetworkX graph
    G_gt = gt.Graph(directed=False)
    nx_to_gt = {}

    for node in G_nx.nodes():
        v = G_gt.add_vertex()
        nx_to_gt[node] = v

    for edge in G_nx.edges():
        v1 = nx_to_gt[edge[0]]
        v2 = nx_to_gt[edge[1]]
        G_gt.add_edge(v1, v2)

    return G_gt, nx_to_gt


def get_two_blocks(state, nx_gt_mapping):

    block_ms = state.get_blocks()
    a, b = list(set(block_ms))

    POLARIZED_GROUP_A = lambda x: block_ms[x] == a
    POLARIZED_GROUP_B = lambda x: block_ms[x] == b

    polarized_group_ms = dict()

    for user_id in nx_gt_mapping:
        if POLARIZED_GROUP_A(nx_gt_mapping[user_id]):
            polarized_group_ms[user_id] = "A"

        elif POLARIZED_GROUP_B(nx_gt_mapping[user_id]):
            polarized_group_ms[user_id] = "B"

        else:
            print("Error")

    return polarized_group_ms


# Model fit functions


def fit_PPSBM(graph, nblocks_min=2, nblocks_max=2):
    state = gt.minimize_blockmodel_dl(
        graph,
        state=gt.PPBlockState,
        multilevel_mcmc_args={
            "B_min": nblocks_min,
            "B_max": nblocks_max,
            "niter": MCMC_ITERATIONS,
        },
    )

    return state, state.entropy()


def run_pipeline(network_name):
    """Run the PP-SBM fitting pipeline for a given network.

    Args:
        network_name (str): Name of the network to process (format: topic_year)
    """
    ensure_dirs()

    # Setup paths
    input_path = os.path.join(NO_OVERLAPS_DIR, f"{network_name}.graphml")
    output_dir = os.path.join(HIERARCHICAL_BATCH_ASSORTATIVE_DIR, network_name)
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Fitting PP-SBM models for {network_name}")

    # Load and prepare graph
    G = load_graph(input_path)
    g, nx_gt_mapping = nx_2_gt(G)

    # Fit models
    for batch_idx in range(N_SAMPLES):
        logging.info(f"Fitting sample {batch_idx + 1}/{N_SAMPLES}")

        try:
            # Fit PP-SBM
            state_lvl1, PP_dl = fit_PPSBM(g, nblocks_min=2, nblocks_max=2)
            logging.info(f"Fitted PP-SBM with description length: {PP_dl}")

            # Get and set group memberships
            group_ms = get_two_blocks(state_lvl1, nx_gt_mapping)
            nx.set_node_attributes(G, group_ms, name="ppsbm")

            # Save results
            G.graph["PP_dl"] = PP_dl
            output_path = os.path.join(
                output_dir, f"{network_name}_{batch_idx}.graphml"
            )
            nx.write_graphml(G, output_path)
            logging.info(f"Saved model to {output_path}")

        except Exception as e:
            logging.error(f"Error in batch {batch_idx}: {e}")


def main():
    """Main execution function."""
    args = parse_args()
    run_pipeline(args.network_name)


if __name__ == "__main__":
    main()
