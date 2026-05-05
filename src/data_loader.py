"""Data loading utilities."""

import pandas as pd


def load_data(path):
    """Load dataset from a CSV file."""
    return pd.read_csv(path)
