from functools import partial
from io import StringIO

from bokeh.models import HTMLTemplateFormatter
import hvplot.pandas
import numpy as np
import pandas as pd
import panel as pn

# CONSTANTS
TITLE = "Database-Name: Material Class 1 Explorer"
DATA_PATH = "https://raw.githubusercontent.com/paolodeangelis/temp_panel/main/data/mclass1.json"
ACCENT = "#3abccd"
PALETTE = ["#50c4d3", "#efa04b", "#f16c90", "#426ba2", "#d7fbc0"]
CATEGORY = "Category 1"
N_ROW = 12
SIDEBAR_W = 380
PLOT_SIZE = [900, 500]


@pn.cache
def initialize_data():
    df = pd.read_json(DATA_PATH)
    df["Ranking"] = 1.0
    df["File"] = df["ID"]
    return df


df = initialize_data()


def min_max_norm(v):
    v_min, v_max = v.min(), v.max()
    return (
        (v - v_min) / (v_max - v_min)
        if v_min != v_max
        else pd.Series(np.zeros(len(v)), index=v.index)
    )


# Define Weights and Sliders
weights = [
    pn.widgets.IntSlider(name=f"Property {i+1}", start=-10, end=10, value=1) for i in range(8)
]
sliders = {
    f"Property {i+1}": pn.widgets.RangeSlider(
        name=f"Property {i+1}", start=df[f"Property {i+1}"].min(), end=df[f"Property {i+1}"].max()
    )
    for i in range(8)
}

select_properties = pn.widgets.MultiChoice(
    name="Database Properties",
    value=["ID", "Property 1", "Property 2", "Ranking"],
    options=df.columns.tolist(),
)
select_ions = pn.widgets.MultiChoice(
    name="Select Categories", value=["A", "B", "C"], options=df[CATEGORY].unique().tolist()
)


# Build Table
def build_interactive_table(*weights, columns, sliders=None, categories=None):
    # Ensure weights are passed as IntSlider objects
    ranking = sum(w.value * min_max_norm(df[f"Property {i+1}"]) for i, w in enumerate(weights))
    df["Ranking"] = min_max_norm(ranking)
    df.sort_values(by="Ranking", ascending=False, inplace=True)

    if sliders:
        for column, slider in sliders.items():
            slider_start, slider_end = slider.value
            df = df[(df[column] >= slider_start) & (df[column] <= slider_end)]  # noqa:F823

    if categories:
        df = df[df[CATEGORY].isin(categories)]

    table = pn.widgets.Tabulator(
        df[columns], pagination="remote", page_size=N_ROW, sizing_mode="stretch_width"
    )
    filename, button = table.download_menu(
        text_kwargs={"name": "Filename", "value": "filtered_data.csv"},
        button_kwargs={"name": "Download"},
    )
    return pn.Column(filename, button, table)


# Build Plot
def build_interactive_plot(*sliders, categories=None):
    plot_df = df.copy()
    for i, slider in enumerate(sliders):
        slider_start, slider_end = slider.value
        plot_df = plot_df[
            (plot_df[f"Property {i+1}"] >= slider_start)
            & (plot_df[f"Property {i+1}"] <= slider_end)
        ]

    if categories:
        plot_df = plot_df[plot_df[CATEGORY].isin(categories)]

    scatter = plot_df.hvplot.scatter(
        x="Property 2", y="Property 3", c=CATEGORY, size=10, cmap=PALETTE, height=PLOT_SIZE[1]
    )
    return scatter


# Bind Interactive Components
downloadable_table = pn.bind(
    build_interactive_table,
    *weights,
    columns=select_properties,
    sliders=sliders,
    categories=select_ions,
)
plot = pn.bind(build_interactive_plot, *sliders.values(), categories=select_ions)

# Layout
sidebar = pn.Column(
    pn.pane.Markdown("## Filters and Controls"),
    *sliders.values(),
    pn.layout.Divider(),
    pn.pane.Markdown("## Ranking"),
    *weights,
)
main = pn.Row(plot, pn.Column(select_properties, downloadable_table))

# Template
template = pn.template.FastListTemplate(
    title=TITLE, sidebar=sidebar, main=main, sidebar_width=SIDEBAR_W, accent=ACCENT
)

# Serve
template.servable()
