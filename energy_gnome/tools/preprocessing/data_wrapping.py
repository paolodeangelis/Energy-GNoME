from typing import Optional

from ase import Atom
from ase.neighborlist import neighbor_list
import numpy as np
import torch
import torch_geometric as tg


def get_encoding() -> tuple[dict[str, int], torch.Tensor, torch.Tensor]:
    """
    Generates encodings for atomic species using atomic numbers and atomic masses.

    This function creates a dictionary mapping atomic symbols to their corresponding
    atomic numbers (zero-based indexing). It also generates two one-hot encoded tensors:
    one for atomic numbers and another for atomic masses.

    Returns:
        tuple[dict[str, int], torch.Tensor, torch.Tensor]: A tuple containing three elements:
            - A dictionary mapping atomic symbols (str) to their corresponding zero-based atomic numbers (int).
            - A one-hot encoded tensor (torch.Tensor) for atomic numbers.
            - A one-hot encoded tensor (torch.Tensor) for atomic masses.
    """
    type_encoding = {}  # Maps atomic symbols to zero-based atomic numbers
    specie_am = []  # List to store atomic masses

    # Populate the type_encoding dictionary and specie_am list
    for Z in range(1, 119):  # Loop over atomic numbers
        specie = Atom(Z)
        type_encoding[specie.symbol] = Z - 1
        specie_am.append(specie.mass)

    # Create one-hot encoded tensors
    type_onehot = torch.eye(len(type_encoding))
    atomicmass_onehot = torch.diag(torch.tensor(specie_am))

    return type_encoding, type_onehot, atomicmass_onehot


def build_data(
    entry,
    type_encoding: dict[str, int],
    type_onehot: torch.Tensor,
    atomicmass_onehot: torch.Tensor,
    y_label: str | None = None,
    r_max: float = 5.0,
    dtype: torch.dtype = torch.float64,
) -> tg.data.Data:
    """
    Constructs a graph data structure from a crystal structure entry.

    This function converts a crystal structure entry into a graph representation suitable for
    machine learning applications. The graph nodes represent atoms, and edges represent atomic
    interactions. Node features include atomic mass and type (one-hot encoded), and edge features
    include interatomic distances and shifts due to periodic boundary conditions.

    Args:
        entry: A crystal structure entry containing atomic positions, symbols, and the unit cell.
        type_encoding (dict[str, int]): Mapping of atomic symbols to their zero-based atomic numbers.
        type_onehot (torch.Tensor): One-hot encoded tensor for atomic types.
        atomicmass_onehot (torch.Tensor): One-hot encoded tensor for atomic masses.
        y_label (str, optional): Label name for the property of interest in the entry.
        r_max (float, optional): Maximum radius to consider for neighbor interactions. Default is 5.0.
        dtype (torch.dtype, optional): Data type for tensors. Default is torch.float64.

    Returns:
        tg.data.Data: A graph data object containing the structure's representation with node and edge features.
    """
    symbols = list(entry.structure.symbols).copy()
    positions = torch.from_numpy(entry.structure.positions.copy())
    lattice = torch.from_numpy(entry.structure.cell.array.copy()).unsqueeze(0)

    # Extract neighbor information
    edge_src, edge_dst, edge_shift = neighbor_list(
        "ijS", a=entry.structure, cutoff=r_max, self_interaction=True
    )

    # Compute edge vectors and lengths
    edge_batch = positions.new_zeros(positions.shape[0], dtype=torch.long)[
        torch.from_numpy(edge_src)
    ]
    edge_vec = (
        positions[torch.from_numpy(edge_dst)]
        - positions[torch.from_numpy(edge_src)]
        + torch.einsum("ni,nij->nj", torch.tensor(edge_shift, dtype=dtype), lattice[edge_batch])
    )
    edge_len = np.around(edge_vec.norm(dim=1).numpy(), decimals=2)

    # Node features
    x = atomicmass_onehot[[type_encoding[specie] for specie in symbols]]
    z = type_onehot[[type_encoding[specie] for specie in symbols]]

    # Construct graph data object
    if y_label:
        target = torch.tensor(getattr(entry, y_label)).unsqueeze(0).unsqueeze(0)
    else:
        target = None
    data = tg.data.Data(
        pos=positions,
        lattice=lattice,
        symbol=symbols,
        x=x,
        z=z,
        edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
        edge_shift=torch.tensor(edge_shift, dtype=dtype),
        edge_vec=edge_vec,
        edge_len=edge_len,
        target=target,
    )

    return data
