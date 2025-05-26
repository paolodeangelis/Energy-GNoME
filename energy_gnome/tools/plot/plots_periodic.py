"""Colored Chemical Periodic Table plot function
(https://github.com/Andrew-S-Rosen/periodic_trends/tree/master)

Author: Andrew S. Rosen
Modified by Paolo De Angelis
"""

import warnings

from ase import Atoms
from ase.data.colors import jmol_colors
from ase.visualize.plot import plot_atoms
from bokeh.io import export_png, export_svg
from bokeh.io import show as show_
from bokeh.models import (
    BasicTicker,
    ColorBar,
    ColumnDataSource,
    LinearColorMapper,
    LogColorMapper,
)
from bokeh.plotting import figure, output_file
from bokeh.sampledata.periodic_table import elements
from bokeh.transform import dodge
from matplotlib.cm import (
    ScalarMappable,
    cividis,
    inferno,
    magma,
    plasma,
    turbo,
    viridis,
)
from matplotlib.colors import LogNorm, Normalize, to_hex
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pandas import DataFrame, options
import torch_geometric as tg


def plotter(
    # filename: str,
    df: DataFrame,
    show: bool = True,
    output_filename: str = None,
    width: int = 1050,
    cmap: str = "plasma",
    alpha: float = 0.65,
    extended: bool = True,
    periods_remove: list[int] = None,
    groups_remove: list[int] = None,
    log_scale: bool = False,
    cbar_height: float = None,
    cbar_standoff: int = 12,
    cbar_fontsize: int = 14,
    blank_color: str = "#c4c4c4",
    under_value: float = None,
    under_color: str = "#140F0E",
    over_value: float = None,
    over_color: str = "#140F0E",
    special_elements: list[str] = None,
    special_color: str = "#6F3023",
) -> figure:
    """
    Plot a heatmap over the periodic table of elements.

    Parameters
    ----------
    df : DataFrame
        Database containing the data to be plotted.
    show : str
        If True, the plot will be shown.
    output_filename : str
        If not None, the plot will be saved to the specified (.html) file.
    width : float
        Width of the plot.
    cmap : str
        plasma, inferno, viridis, magma, cividis, turbo
    alpha : float
        Alpha value (transparency).
    extended : bool
        If True, the lanthanoids and actinoids will be shown.
    periods_remove : List[int]
        Period numbers to be removed from the plot.
    groups_remove : List[int]
        Group numbers to be removed from the plot.
    log_scale : bool
        If True, the colorbar will be logarithmic.
    cbar_height : int
        Height of the colorbar.
    cbar_standoff : int
        Distance between the colorbar and the plot.
    cbar_fontsize : int
        Fontsize of the colorbar label.
    blank_color : str
        Hexadecimal color of the elements without data.
    under_value : float
        Values <= under_value will be colored with under_color.
    under_color : str
        Hexadecimal color to be used for the lower bound color.
    over_value : float
        Values >= over_value will be colored with over_color.
    under_color : str
        Hexadecial color to be used for the upper bound color.
    special_elements: List[str]
        List of elements to be colored with special_color.
    special_color: str
        Hexadecimal color to be used for the special elements.

    Returns
    -------
    figure
        Bokeh figure object.
    """

    options.mode.chained_assignment = None

    # Assign color palette based on input argument
    if cmap == "plasma":
        cmap = plasma
        bokeh_palette = "Plasma256"
    elif cmap == "inferno":
        cmap = inferno
        bokeh_palette = "Inferno256"
    elif cmap == "magma":
        cmap = magma
        bokeh_palette = "Magma256"
    elif cmap == "viridis":
        cmap = viridis
        bokeh_palette = "Viridis256"
    elif cmap == "cividis":
        cmap = cividis
        bokeh_palette = "Cividis256"
    elif cmap == "turbo":
        cmap = turbo
        bokeh_palette = "Turbo256"
    else:
        ValueError("Invalid color map.")

    # Define number of and groups
    period_label = ["1", "2", "3", "4", "5", "6", "7"]
    group_range = [str(x) for x in range(1, 19)]

    # Remove any groups or periods
    if groups_remove:
        for gr in groups_remove:
            gr = gr.strip()
            group_range.remove(str(gr))
    if periods_remove:
        for pr in periods_remove:
            pr = pr.strip()
            period_label.remove(str(pr))

    # Read in data from CSV file
    data_elements = []
    data_list = []
    # for row in reader(open(filename)):
    #     data_elements.append(row[0])
    #     data_list.append(row[1])
    data_elements = df["element"].tolist()
    data_list = df["count"].tolist()
    data = [float(i) for i in data_list]

    if len(data) != len(data_elements):
        raise ValueError("Unequal number of atomic elements and data points")

    period_label.append("blank")
    period_label.append("La")
    period_label.append("Ac")

    if extended:
        count = 0
        for i in range(56, 70):
            elements.period.loc[i] = "La"
            elements.group.loc[i] = str(count + 4)  # this was raising a warning without .loc
            count += 1

        count = 0
        for i in range(88, 102):
            elements.period.loc[i] = "Ac"
            elements.group.loc[i] = str(count + 4)
            count += 1

    # Define matplotlib and bokeh color map
    if log_scale:
        for datum in data:
            if datum < 0:
                raise ValueError(f"Entry for element {datum} is negative but log-scale is selected")
        color_mapper = LogColorMapper(palette=bokeh_palette, low=min(data), high=max(data))
        norm = LogNorm(vmin=min(data), vmax=max(data))
    else:
        color_mapper = LinearColorMapper(palette=bokeh_palette, low=min(data), high=max(data))
        norm = Normalize(vmin=min(data), vmax=max(data))
    color_scale = ScalarMappable(norm=norm, cmap=cmap).to_rgba(data, alpha=None)

    # Set blank color
    color_list = [blank_color] * len(elements)

    # Compare elements in dataset with elements in periodic table
    for i, data_element in enumerate(data_elements):
        element_entry = elements.symbol[elements.symbol.str.lower() == data_element.lower()]
        if element_entry.empty is False:
            element_index = element_entry.index[0]
        else:
            warnings.warn("Invalid chemical symbol: " + data_element)
        if color_list[element_index] != blank_color:
            warnings.warn("Multiple entries for element " + data_element)
        elif under_value is not None and data[i] <= under_value:
            color_list[element_index] = under_color
        elif over_value is not None and data[i] >= over_value:
            color_list[element_index] = over_color
        else:
            color_list[element_index] = to_hex(color_scale[i])

    if special_elements:
        for k, v in elements["symbol"].iteritems():
            if v in special_elements:
                color_list[k] = special_color

    # Define figure properties for visualizing data
    source = ColumnDataSource(
        data=dict(
            group=[str(x) for x in elements["group"]],
            period=[str(y) for y in elements["period"]],
            sym=elements["symbol"],
            atomic_number=elements["atomic number"],
            type_color=color_list,
        )
    )

    # Plot the periodic table
    p = figure(x_range=group_range, y_range=list(reversed(period_label)), tools="save")
    p.width = width
    p.outline_line_color = None
    p.background_fill_color = "#FFFFFF"  # None
    p.border_fill_color = None
    p.toolbar_location = "above"
    p.rect("group", "period", 0.9, 0.9, source=source, alpha=alpha, color="type_color")
    p.axis.visible = False
    text_props = {
        "source": source,
        "angle": 0,
        "color": "black",
        "text_align": "left",
        "text_baseline": "middle",
    }
    x = dodge("group", -0.4, range=p.x_range)
    y = dodge("period", 0.3, range=p.y_range)
    p.text(
        x=x,
        y="period",
        text="sym",
        text_font_style="bold",
        text_font_size="16pt",
        **text_props,
    )
    p.text(x=x, y=y, text="atomic_number", text_font_size="11pt", **text_props)

    color_bar = ColorBar(
        color_mapper=color_mapper,
        ticker=BasicTicker(desired_num_ticks=10),
        border_line_color=None,
        label_standoff=cbar_standoff,
        location=(0, 0),
        orientation="vertical",
        scale_alpha=alpha,
        major_label_text_font_size=f"{cbar_fontsize}pt",
    )
    color_bar.background_fill_alpha = 1.0
    color_bar.background_fill_color = "#FFFFFF"

    if cbar_height is not None:
        color_bar.height = cbar_height

    p.add_layout(color_bar, "right")
    p.grid.grid_line_color = None

    if output_filename:
        ext = output_filename.split(".")[-1].lower()
        if ext == "html":
            p.output_backend = "svg"
            output_file(output_filename)
        elif ext == "png":
            export_png(p, filename=output_filename)
        elif ext == "svg":
            p.output_backend = "svg"
            export_svg(p, filename=output_filename)
        else:
            raise ValueError(f"Not {ext} file exporter")

    if show:
        show_(p)

    return p


def plot_example(df, i=12, label_edges=False, figsize=(14, 10)):
    # plot an example crystal structure and graph
    entry = df.iloc[i]["data"]

    # get graph with node and edge attributes
    g = tg.utils.to_networkx(entry, node_attrs=["symbol"], edge_attrs=["edge_len"], to_undirected=True)

    # remove self-loop edges for plotting
    g.remove_edges_from(list(nx.selfloop_edges(g)))
    node_labels = dict(zip([k[0] for k in g.nodes.data()], [k[1]["symbol"] for k in g.nodes.data()]))
    edge_labels = dict(zip([(k[0], k[1]) for k in g.edges.data()], [k[2]["edge_len"] for k in g.edges.data()]))

    # project positions of nodes to 2D for plotting
    pos = dict(zip(list(g.nodes), [np.roll(k, 2)[:-1][::-1] for k in entry.pos.numpy()]))

    # plot unit cell
    fig, ax = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2, 3]})
    atoms = Atoms(
        symbols=entry.symbol,
        positions=entry.pos.numpy(),
        cell=entry.lattice.squeeze().numpy(),
        pbc=True,
    )
    plot_atoms(atoms, ax[0], radii=0.25, rotation=("0x,90y,0z"))

    # plot graph
    color = [jmol_colors[i] for i in atoms.get_atomic_numbers()]
    nx.draw_networkx(
        g,
        ax=ax[1],
        labels=node_labels,
        pos=pos,
        node_size=500,
        node_color=color,
        edge_color="gray",
    )

    if label_edges:
        nx.draw_networkx_edge_labels(g, ax=ax[1], edge_labels=edge_labels, pos=pos, label_pos=0.5)

    # format axes
    ax[0].set_xlabel(r"$x_1\ (\AA)$")
    ax[0].set_ylabel(r"$x_2\ (\AA)$")
    ax[0].set_title("Crystal structure")
    ax[1].set_aspect("equal")
    ax[1].axis("off")
    ax[1].set_title("Crystal graph")
    pad = np.array([-0.5, 0.5])
    ax[1].set_xlim(np.array(ax[1].get_xlim()) + pad)
    ax[1].set_ylim(np.array(ax[1].get_ylim()) + pad)
    fig.subplots_adjust(wspace=0.4)
