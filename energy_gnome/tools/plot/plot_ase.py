from ase import Atoms
from ase.data import chemical_symbols
from ase.data.colors import jmol_colors
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import numpy as np


def plot_atoms_legend(atoms: Atoms, ax: Axes, **kwargs) -> None:
    """
    Plots a legend on a given Axes object to represent unique atoms in an ASE Atoms object.

    This function creates a legend where each unique atom type in the Atoms object is represented
    by a colored marker. The color and label of each marker correspond to the atom's type.

    Args:
        atoms (Atoms): An ASE Atoms object containing the atomic structures.
        ax (Axes): A Matplotlib Axes object where the legend will be plotted.

    Keyword Args:
        **kwargs: Additional keyword arguments are passed to the Axes.legend method.

    Returns:
        None: The function adds a legend to the given Axes object but does not return anything.
    """
    # Extract unique atom numbers and corresponding colors and labels
    unique_atom_numbers = np.unique(atoms.numbers)
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markeredgecolor="k",
            label=chemical_symbols[number],
            markerfacecolor=jmol_colors[number],
            markersize=12,
        )
        for number in unique_atom_numbers
    ]

    try:
        kwargs["loc"]
    except KeyError:
        kwargs["loc"] = "upper right"
    # Add the legend to the provided Axes object
    ax.legend(handles=legend_elements, **kwargs)
