import os
import logging
import pandas as pd
import networkx as nx
from config import (
    DATA_DIR,
    OVERLAPS_DIR,
    NO_OVERLAPS_DIR,
    TOPICS,
    ELECTION_PERIODS,
    ensure_dirs,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_data():
    """Load and preprocess the retweet data."""
    parquet_file_path = os.path.join(DATA_DIR, "relevant_retweets.parquet")
    logging.info(f"Loading data from {parquet_file_path}")
    df = pd.read_parquet(parquet_file_path)

    # Process each year's data
    dfs = {}
    for year in ["19", "23"]:
        period = ELECTION_PERIODS[year]
        start_date = pd.Timestamp(period["start"], tz="UTC")
        end_date = pd.Timestamp(period["end"], tz="UTC")

        dfs[year] = df[
            (start_date < df["timestamp"]) & (df["timestamp"] <= end_date)
        ].reset_index(drop=True)

    return dfs["19"], dfs["23"]


def create_topic_networks(df19, df23):
    """Create topic-specific networks for both regular and non-overlapping cases."""
    ensure_dirs()  # Ensure all directories exist

    for topic in TOPICS:
        logging.info(f"Processing topic: {topic}")

        # Create regular networks
        for year, df in [("19", df19), ("23", df23)]:
            # Filter data for topic
            topic_df = df[df[topic] == 1].reset_index(drop=True)

            # Create network
            G = nx.from_pandas_edgelist(
                topic_df,
                source="retweeter",
                target="retweeted",
                create_using=nx.MultiDiGraph(),
            )

            # Save regular network
            output_path = os.path.join(OVERLAPS_DIR, f"{topic}_{year}.graphml")
            nx.write_graphml(G, output_path)
            logging.info(f"Saved regular network to {output_path}")

            # Create strict network (no overlapping topics)
            topic_df_strict = topic_df[topic_df[TOPICS].sum(axis=1) == 1].reset_index(
                drop=True
            )

            G_strict = nx.from_pandas_edgelist(
                topic_df_strict,
                source="retweeter",
                target="retweeted",
                create_using=nx.MultiDiGraph(),
            )

            # Save strict network
            output_path = os.path.join(NO_OVERLAPS_DIR, f"{topic}_{year}.graphml")
            nx.write_graphml(G_strict, output_path)
            logging.info(f"Saved non-overlapping network to {output_path}")


def main():
    """Main execution function."""
    df19, df23 = load_data()
    create_topic_networks(df19, df23)


if __name__ == "__main__":
    main()
