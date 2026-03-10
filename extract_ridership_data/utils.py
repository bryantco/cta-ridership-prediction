"""Utility functions for extracting CTA data from Socrata."""

from itertools import islice

import pandas as pd
from sodapy import Socrata


def fetch_socrata_dataset(dataset_id: str, chunk_size: int = 10_000) -> pd.DataFrame:
    """Download an entire Socrata dataset by ID and return it as a DataFrame.

    Parameters
    ----------
    dataset_id:
        The four-by-four Socrata dataset identifier (e.g. ``'5neh-572f'``).
    chunk_size:
        Number of records to fetch per iteration.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing all records from the dataset.
    """
    client = Socrata(
        "data.cityofchicago.org",
        timeout=1000,
        app_token=None,
    )

    data_iter = client.get_all(dataset_id)

    chunks = []
    while True:
        chunk = list(islice(data_iter, chunk_size))
        print("Grabbing chunk of data...")
        if not chunk:
            break
        chunks.append(pd.DataFrame(chunk))

    return pd.concat(chunks, ignore_index=True)
