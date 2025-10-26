import os
import logging
import numpy as np
import networkx as nx

from config import (
    TOPICS,
    YEARS,
    HIERARCHICAL_BATCH_ASSORTATIVE_DIR,
    BEST_ASSORTATIVE_DIR,
    N_SAMPLES,
    ensure_dirs,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_networks():
    """Generate list of networks to process."""
    return [f"{topic}_{year}" for topic in TOPICS for year in YEARS]


def select_best_partition(network_name):
    """Select the best partition based on description length.

    Args:
        network_name (str): Name of the network to process

    Returns:
        tuple: (best_network, best_dl, best_idx)
    """
    input_dir = os.path.join(HIERARCHICAL_BATCH_ASSORTATIVE_DIR, network_name)
    best_dl = np.inf
    best_idx = 0

    # Find best partition
    for sample_idx in range(N_SAMPLES):
        sample_path = os.path.join(input_dir, f"{network_name}_{sample_idx}.graphml")

        try:
            G = nx.read_graphml(sample_path)
            candidate_dl = G.graph["PP_dl"]

            logging.info(f"Sample {sample_idx}: DL = {candidate_dl}")

            if candidate_dl < best_dl:
                logging.info(f"Found better fit: DL = {candidate_dl}")
                best_dl = candidate_dl
                best_idx = sample_idx

        except Exception as e:
            logging.error(f"Error processing sample {sample_idx}: {e}")

    # Load and return best network
    best_path = os.path.join(input_dir, f"{network_name}_{best_idx}.graphml")
    best_network = nx.read_graphml(best_path)

    return best_network, best_dl, best_idx


def save_best_partition(network_name, network, best_dl):
    """Save the best partition for a network.

    Args:
        network_name (str): Name of the network
        network (nx.Graph): Best network partition
        best_dl (float): Description length of best partition
    """
    output_dir = os.path.join(BEST_ASSORTATIVE_DIR, network_name)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{network_name}.graphml")
    nx.write_graphml(network, output_path)
    logging.info(f"Saved best partition (DL={best_dl}) to {output_path}")


def main():
    """Main execution function."""
    ensure_dirs()
    networks = get_networks()

    for network_name in networks:
        logging.info(f"Processing {network_name}")

        try:
            best_network, best_dl, best_idx = select_best_partition(network_name)
            save_best_partition(network_name, best_network, best_dl)
            logging.info(
                f"Completed {network_name} (best_idx={best_idx}, DL={best_dl})"
            )

        except Exception as e:
            logging.error(f"Error processing {network_name}: {e}")


if __name__ == "__main__":
    main()
