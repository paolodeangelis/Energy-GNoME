import numpy as np
import pandas as pd


def get_neighbors(df: pd.DataFrame, idx: list[int]) -> np.ndarray:
    """
    Calculates the number of neighbors for each atom in a subset of entries from a DataFrame.

    This function processes a DataFrame containing entries with graph data (representing atomic structures).
    It computes the number of neighbors for each atom in the structures indexed by 'idx'.

    Args:
        df (pd.DataFrame): The DataFrame containing graph data of atomic structures. Each entry is expected
                           to have an attribute 'data' with 'pos' and 'edge_index'.
        idx (list[int]): A list of indices indicating the entries in the DataFrame to be processed.

    Returns:
        np.ndarray: An array containing the number of neighbors for each atom in the specified entries.
    """
    neighbors_count = []

    # Iterate over each specified entry in the DataFrame
    for entry in df.iloc[idx].itertuples():
        N = entry.data.pos.shape[0]  # Number of atoms in the entry

        # Count neighbors for each atom
        for i in range(N):
            count = len((entry.data.edge_index[0] == i).nonzero())
            neighbors_count.append(count)

    return np.array(neighbors_count)
