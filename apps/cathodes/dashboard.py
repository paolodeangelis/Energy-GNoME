from functools import partial
from io import StringIO
from itertools import product

from bokeh.models import HoverTool, HTMLTemplateFormatter
import hvplot.pandas
import numpy as np
import pandas as pd
import panel as pn
import requests

# CONSTANTS (settings)
TITLE = "Database-Name: Material class 1 explorer"
# DATA_PATH = "https://raw.githubusercontent.com/paolodeangelis/temp_panel/main/data/mclass1.json"
DATA_PATH_TEMPLATE = "https://raw.githubusercontent.com/paolodeangelis/Energy-GNoME/main/data/final/cathodes/{ctype}/{ion}/candidates.json"
BIB_FILE = (
    "https://raw.githubusercontent.com/paolodeangelis/temp_panel/main/assets/gnome-energy.bib"
)
RIS_FILE = (
    "https://raw.githubusercontent.com/paolodeangelis/temp_panel/main/assets/gnome-energy.ris"
)
ACCENT = "#3abccd"
PALETTE = [
    "#50c4d3",
    "#efa04b",
    "#f16c90",
    "#426ba2",
    "#d7fbc0",
    "#ffd29b",
    "#fe8580",
    "#009b8f",
    "#73bced",
]
WORKING_IONS = ["Li", "Na", "Mg", "K", "Ca", "Cs"]
WORKING_IONS_ACTIVE = ["Li", "Na", "Mg"]
CATHODE_TYPE = ["insertion"]
CATEGORY = "Working Ion"
CATEGORY_ACTIVE = WORKING_IONS_ACTIVE
COLUMNS = [
    "Material Id",
    "Composition",
    "Crystal System",
    "Formation Energy (eV/atom)",
    "Formula",
    "Average Voltage (V)",
    "Average Voltage (deviation) (V)",
    "AI-experts confidence (-)",
    "AI-experts confidence (deviation) (-)",
    "Max Volume expansion (%)",
    "Max Volume expansion (deviation) (%)",
    "Stability charge (eV/atom)",
    "Stability charge (deviation) (eV/atom)",
    "Stability discharge (eV/atom)",
    "Stability discharge (deviation) (eV/atom)",
    "Volumetric capacity (mAh/L)",
    "Gravimetric capacity (mAh/g)",
    "Volumetric energy (Wh/L)",
    "Gravimetric energy (Wh/kg)",
    "Ranking",
    "File",
]
HOVER_COL =  [
    ("Material Id", "@{Material Id}"),
    ("Working Ion", "@{Working Ion}"),
    ("Average Voltage", "@{Average Voltage (V)}{0.2f} V"),
    ("AI-experts confidence", "@{AI-experts confidence (-)}{0.2f}"),
    ("Max Volume expansion", "@{Max Volume expansion (%)}{0.2f} %"),  
    ("Volumetric capacity", "@{Volumetric capacity (mAh/L)}{0.2f} mAh/L"),  
    ("Gravimetric capacity", "@{Gravimetric capacity (mAh/g)}{0.2f} mAh/g"), 
    ("Volumetric energy", "@{Volumetric energy (Wh/L)}{0.2f} Wh/L"),   
    ("Gravimetric energy", "@{Gravimetric energy (Wh/kg)}{0.2f} Wh/kg"),     
]

COLUMNS_ACTIVE = [
    "Material Id",
    "Composition",
    "Average Voltage (V)",
    "AI-experts confidence (-)",
    # "Max Volume expansion (%)",
    # "Stability charge (eV/atom)",
    # "File",
]
N_ROW = 12
SIDEBAR_W = 380
SIDEBAR_WIDGET_W = 320
PLOT_SIZE = [900, 500]  # WxH
TABLE_FORMATTER = {
    "File": HTMLTemplateFormatter(
        template=r'<code><a href="https://raw.githubusercontent.com/paolodeangelis/temp_panel/main/data/cif/test1.cif?download=1" download="<%= value %>.cif" target="_blank"> <i class="fas fa-external-link-alt"></i> <%= value %>.cif </a></code>'  # noqa: E501, W505
    )
    # HTMLTemplateFormatter(template=r'<code><a href="file:///C:/Users/Paolo/OneDrive%20-%20Politecnico%20di%20Torino/3-Articoli/2024-GNoME/plots/<%= value %>.cif?download=1" download="realname.cif" > <%= value %>.cif </a></code>') # noqa: E501, W505
}
ABOUT_W = 500
ABOUT_MSG = """
# About
Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Proin id porttitor dui. In neque lectus, malesuada sed arcu vitae, cursus tincidunt nisl. Etiam lacinia congue porttitor. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Donec mollis id justo eu mattis. Duis quis vulputate massa. Morbi est tortor, fermentum in neque non, aliquet suscipit justo. Sed sed odio efficitur, viverra ante fermentum, pulvinar ex. Sed id bibendum elit, faucibus convallis dui. Donec eu pulvinar orci.
Nullam et libero vitae orci molestie gravida at nec risus. Nam ipsum sapien, lacinia molestie nulla quis, ornare laoreet velit. Maecenas nec volutpat nulla. Ut erat ipsum, porttitor vel bibendum in, volutpat in ex. Vestibulum vel odio orci.
Proin eget turpis et erat faucibus feugiat non vel nisi. Ut sed erat sed ligula cursus bibendum. Proin ultricies accumsan diam, vitae fermentum nulla commodo in. Nulla vehicula odio sit amet dictum tristique. Phasellus non posuere mi, vel vehicula neque. Donec leo turpis, iaculis vel enim eget, convallis elementum mi. Donec euismod mattis orci et interdum.

If you find this dataset valuable, please consider citing the original work:

> De Angelis, Paolo, et al. "Article Title to be defiened" *Journal*.

"""
FOOTER = f"""
<link rel="stylesheet"
href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/dist/tabler-icons.min.css" />

<footer style="color: #777; padding: 0px; text-align: center;
line-height: 1.0; width: 100%; margin-left: auto; margin-right: auto;">
    <p style="font-size: 1.2em; margin-top: 0px; margin-bottom: 0.2em;">
        <a href="https://paolodeangelis.github.io/" style="color: #777;
        text-decoration: none;" target="_blank" >
            <i class="ti ti-code"
            style="vertical-align: middle; font-size: 1.25em;"></i>
            with
            <i class="ti ti-heart-filled"
            style="vertical-align: middle; font-size: 1.25em;"></i>
            by <strong style="color: #555"
            onmouseover="this.style.color='{ACCENT}'"
            onmouseout="this.style.color='#555'">
            Paolo De Angelis
            </strong>
        </a>
    </p>
    <p style="font-size: 0.85em; margin-top: 0px">
        Made entirely with OSS packages:
        <a href="https://panel.holoviz.org/"
        style="color: #555; text-decoration: none;"
        target="_blank"
        onmouseover="this.style.color='{ACCENT}'"
        onmouseout="this.style.color='#555'">
        <strong>Panel</strong>
        </a>,
        <a href="https://holoviews.org/"
        style="color: #555; text-decoration: none;"
        target="_blank" onmouseover="this.style.color='{ACCENT}'"
        onmouseout="this.style.color='#555'">
        <strong>Holoviews</strong>
        </a>,
        <a href="https://bokeh.org/"
        style="color: #555; text-decoration: none;"
        target="_blank"
        onmouseover="this.style.color='{ACCENT}'"
        onmouseout="this.style.color='#555'">
        <strong>Bokeh</strong>
        </a>,
        <a href="https://pandas.pydata.org/"
        style="color: #555; text-decoration: none;" target="_blank"
        onmouseover="this.style.color='{ACCENT}'"
        onmouseout="this.style.color='#555'">
        <strong>Pandas</strong>
        </a>,
        <a href="https://numpy.org/"
        style="color: #555; text-decoration: none;" target="_blank"
        onmouseover="this.style.color='{ACCENT}'"
        onmouseout="this.style.color='#555'">
        <strong>Numpy</strong>
        </a>.
    </p>

</footer>
"""

# Database
global df


@pn.cache
def initialize_data() -> pd.DataFrame:
    """
    Load and initialize datasets for different working ions efficiently for use in a Panel dashboard.

    This function dynamically loads datasets for each working ion, merges them,
    and adds a column specifying the working ion. It ensures efficient memory usage and fast execution.

    Returns:
        pd.DataFrame: The merged DataFrame with additional columns and the working ion specified.
    """

    # Use a generator to load and process data lazily
    def load_and_process(ion, ctype):
        path = DATA_PATH_TEMPLATE.format(ctype=ctype, ion=ion)
        df = pd.read_json(path)
        df["Cathode Type"] = ctype
        df["Working Ion"] = ion
        df["Ranking"] = 1.0
        df["File"] = df["Material Id"]
        # Downcast float64 to float32 for memory efficiency
        float_cols = df.select_dtypes(include=["float64"]).columns
        df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast="float")
        return df

    # Merge datasets for all ions
    merged_df = pd.concat(
        (load_and_process(ion, ctype) for ctype, ion in product(CATHODE_TYPE, WORKING_IONS)),
        ignore_index=True,
    )

    return merged_df


df = initialize_data()


global table
table = pn.widgets.Tabulator(
    df,
    pagination="remote",
    page_size=N_ROW,
    sizing_mode="stretch_width",
    formatters=TABLE_FORMATTER,
)

global all_columns
all_columns = df.columns.unique()

global all_ions
all_ions = WORKING_IONS  # df[CATEGORY].unique()


# Functions
def get_raw_file_github(url: str) -> StringIO:
    """
    Fetches the raw content of a file from a given GitHub URL and returns it as a StringIO object.

    Args:
        url (str): The URL of the raw file on GitHub.

    Returns:
        StringIO: A StringIO object containing the content of the file.

    Raises:
        requests.HTTPError: If the HTTP request returned an unsuccessful status code.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        output = StringIO(
            response.text
        )  # Use response.text to directly decode and write to StringIO
        return output
    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        raise http_err  # Re-raise the exception after logging it
    except Exception as err:
        print(f"An error occurred: {err}")
        raise err  # Re-raise the exception after logging it
    finally:
        response.close()  # Ensure the response is always closed properly


def apply_range_filter(
    df: pd.DataFrame, column: str, value_range: pn.widgets.RangeSlider
) -> pd.DataFrame:
    """
    Apply a range filter to a specified column in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to filter.
        column (str): The column name on which to apply the filter.
        value_range (pn.widgets.RangeSlider): A Panel RangeSlider widget or a tuple indicating the range values.

    Returns:
        pd.DataFrame: The filtered DataFrame containing rows within the specified range.
    """
    start, end = value_range if isinstance(value_range, tuple) else value_range.value
    return df[(df[column] >= start) & (df[column] <= end)]


def apply_category_filter(df: pd.DataFrame, category: str, item_to_hide: str) -> pd.DataFrame:
    """
    Filter out rows from the DataFrame where the specified category column matches the item to hide.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        category (str): The column name in the DataFrame to apply the filter on.
        item_to_hide (str): The value in the category column that should be excluded from the result.

    Returns:
        pd.DataFrame: A DataFrame with rows where the category column does not match the item_to_hide.
    """
    if category not in df.columns:
        raise ValueError(f"Column '{category}' not found in DataFrame.")

    return df[df[category] != item_to_hide]


def create_range_slider(col: str, name: str = "filter") -> pn.widgets.RangeSlider:
    """
    Create a RangeSlider widget for filtering a DataFrame column.

    Args:
        col (str): The name of the column to apply the RangeSlider on.
        name (str, optional): The name of the slider widget. Defaults to 'filter'.

    Returns:
        pn.widgets.RangeSlider: A RangeSlider widget initialized with the column's value range.
    """
    slider = pn.widgets.RangeSlider(
        start=table.value[col].min(),
        end=table.value[col].max(),
        name=name,
        sizing_mode="fixed",
        width=SIDEBAR_WIDGET_W,
    )
    return slider


def min_max_norm(v: pd.Series) -> pd.Series:
    """
    Normalize a Pandas Series using min-max normalization.

    Args:
        v (pd.Series): The Series to be normalized.

    Returns:
        pd.Series: The min-max normalized Series, where values are scaled to the range [0, 1].
    """
    v_min = v.min()
    v_max = v.max()
    if v_min == v_max:
        return pd.Series(
            np.zeros(len(v)), index=v.index
        )  # Return a Series of zeros if all values are the same
    return (v - v_min) / (v_max - v_min)


def show_selected_columns(table: pn.widgets.Tabulator, columns: list) -> pn.widgets.Tabulator:
    """
    Update the table widget to display only the selected columns by hiding the others.

    Args:
        table (pn.widgets.Tabulator): The table widget.
        columns (list): A list of column names to be displayed.

    Returns:
        pn.widgets.Tabulator: The updated table widget with only the selected columns visible.
    """
    hidden_columns = set(all_columns) - set(columns)
    table.hidden_columns = list(hidden_columns)
    return table


def build_interactive_table(
    # weights
    w_property1: pn.widgets.IntSlider,
    w_property2: pn.widgets.IntSlider,
    w_property3: pn.widgets.IntSlider,
    w_property4: pn.widgets.IntSlider,
    w_property5: pn.widgets.IntSlider,
    w_property6: pn.widgets.IntSlider,
    w_property7: pn.widgets.IntSlider,
    w_property8: pn.widgets.IntSlider,
    w_property9: pn.widgets.IntSlider,
    w_property10: pn.widgets.IntSlider,
    w_property11: pn.widgets.IntSlider,
    w_property12: pn.widgets.IntSlider,
    w_property13: pn.widgets.IntSlider,
    w_property14: pn.widgets.IntSlider,
    # sliders
    # s_classifier_mean: pn.widgets.RangeSlider,
    columns: list,
    sliders: dict = None,
    categories: list = None,
) -> pn.Column:
    """
    Build an interactive table with ranking and filtering features based on the provided weights and filters.

    Args:
        w_property1 to w_property8: IntSlider widgets representing weights for each property.
        columns (list): A list of column names to be displayed in the table.
        sliders (dict, optional): A dictionary where keys are column names and values are
                                  RangeSlider widgets for filtering. Defaults to None.
        categories (list, optional): A list of categories to be displayed. Defaults to None.

    Returns:
        pn.Column: A Panel Column containing the filename input, download button, and the interactive table.
    """
    # Calculate ranking based on weights and normalize
    ranking = (
        w_property1 * min_max_norm(df["Average Voltage (V)"])
        + w_property2 * min_max_norm(df["Average Voltage (deviation) (V)"])
        + w_property3 * min_max_norm(df["AI-experts confidence (-)"])
        + w_property4 * min_max_norm(df["AI-experts confidence (deviation) (-)"])
        + w_property5 * min_max_norm(df["Max Volume expansion (%)"])
        + w_property6 * min_max_norm(df["Max Volume expansion (deviation) (%)"])
        + w_property7 * min_max_norm(df["Stability charge (eV/atom)"])
        + w_property8 * min_max_norm(df["Stability charge (deviation) (eV/atom)"])
        + w_property9 * min_max_norm(df["Stability discharge (eV/atom)"])
        + w_property10 * min_max_norm(df["Stability discharge (deviation) (eV/atom)"])
        + w_property11 * min_max_norm(df["Volumetric capacity (mAh/L)"])
        + w_property12 * min_max_norm(df["Gravimetric capacity (mAh/g)"])
        + w_property13 * min_max_norm(df["Volumetric energy (Wh/L)"])
        + w_property14 * min_max_norm(df["Gravimetric energy (Wh/kg)"])
    )
    # Add the ranking to the DataFrame and normalize
    df["Ranking"] = min_max_norm(ranking)
    # Sort DataFrame by Ranking
    df.sort_values(by="Ranking", ascending=False, inplace=True, ignore_index=True)
    # Create a Tabulator widget with the sorted DataFrame
    table = pn.widgets.Tabulator(
        df,
        pagination="remote",
        page_size=N_ROW,
        sizing_mode="stretch_width",
        formatters=TABLE_FORMATTER,
    )
    # Show selected columns
    table = show_selected_columns(table, columns)
    # Apply range filters using sliders
    # table.add_filter(pn.bind(apply_range_filter, column="classifier_mean", value_range=s_classifier_mean))
    if sliders:
        for column, slider in sliders.items():
            if column in df.columns:  # Ensure the column exists in the DataFrame
                table.add_filter(pn.bind(apply_range_filter, column=column, value_range=slider))
    # Apply category filters for categories
    if categories:
        hidden_ions = set(all_ions) - set(categories)
        for ion in hidden_ions:
            table.add_filter(pn.bind(apply_category_filter, category=CATEGORY, item_to_hide=ion))
    # Add download section
    filename, button = table.download_menu(
        text_kwargs={"name": "Enter filename", "value": "cathode_candidates.csv"},
        button_kwargs={"name": "Download table"},
    )
    return pn.Column(filename, button, table)


# hover = HoverTool(
#     # tooltips=[
#     #     ("ID", "@{ID}"),
#     #     ("Category 1", "@{Category 1}"),
#     #     ("Average Voltage (V)", "@{Property 1}{0.2f}"),
#     #     ("AI-experts confidence (-)", "@{Property 2}{0.2f}"),
#     #     ("Max Volume expansion (%)", "@{Property 3}{0.2f}"),
#     # ],
#     tooltips=[
#             (
#                 col,
#                 (
#                     f"@{{{col}}}{{0.2f}}"
#                     if df[col].dtype in ["float64", "float32", "float", "int"]
#                     else f"@{{{col}}}"
#                 ),
#             )
#             for col in df.columns
#         ],
#      mode='mouse'  # 'mouse' mode ensures only the topmost point is selected
# )


def build_interactive_plot(
    s_property1: pn.widgets.RangeSlider,
    s_property2: pn.widgets.RangeSlider,
    s_property3: pn.widgets.RangeSlider,
    s_property4: pn.widgets.RangeSlider,
    s_property5: pn.widgets.RangeSlider,
    s_property6: pn.widgets.RangeSlider,
    s_property7: pn.widgets.RangeSlider,
    s_property8: pn.widgets.RangeSlider,
    categories: list = None,
) -> hvplot:
    """
    Builds an interactive scatter plot based on selected filters and ion selection.

    Args:
        s_property1-8 (pn.widgets.RangeSlider): Property filters.
        categories (list, optional): List of categories to include in the plot. Default is None.

    Returns:
        hvplot: An interactive scatter plot with applied filters.
    """
    plot_df = df.copy()

    # Apply filters based on sliders
    filters = [
        ("Average Voltage (V)", s_property1),
        ("AI-experts confidence (-)", s_property2),
        ("Max Volume expansion (%)", s_property3),
        ("Stability charge (eV/atom)", s_property4),
        ("Stability discharge (eV/atom)", s_property5),
        ("Volumetric capacity (mAh/L)", s_property6),
        ("Gravimetric capacity (mAh/g)", s_property7),
        ("Volumetric energy (Wh/L)", s_property8),
    ]

    for col, slider in filters:
        plot_df = plot_df[(plot_df[col] >= slider[0]) & (plot_df[col] <= slider[1])]

    # Apply ion filter if provided
    if categories:
        plot_df = plot_df[plot_df[CATEGORY].isin(categories)]

    # Background scatter plot with all data
    back_scatter = df.hvplot.scatter(
        x="AI-experts confidence (-)",
        y="Max Volume expansion (%)",
        s=100,
        alpha=0.25,
        color="#444",
        line_color="white",
        hover_cols="all",
    ).opts(
        tools=[],
        logx=False,
        logy=True,
        xlabel="Property 2 (-)",
        ylabel="Property 3 (-)",
    )

    # Foreground scatter plot with filtered data
    front_scatter = plot_df.hvplot.scatter(
        x="AI-experts confidence (-)",
        y="Max Volume expansion (%)",
        s=100,
        line_color="white",
        c=CATEGORY,
        legend="top",
        hover_cols="all",
    ).opts(
        logx=False,
        logy=True,
        xlabel="Property 2 (-)",
        ylabel="Property 3 (-)",
        cmap=PALETTE,
        # tools=[hover],
        hover_tooltips=HOVER_COL
    )

    # Combine background and foreground scatter plots
    scatter = back_scatter * front_scatter
    scatter.opts(
        min_width=PLOT_SIZE[0],
        height=PLOT_SIZE[1],
        show_grid=True,
    )

    return scatter


# (1) Widget SIDEBAR : properties weights
weights = {}
weights_helper = {}
# Property 1
w_property1 = pn.widgets.IntSlider(
    name="Average Voltage (V)",
    start=-10,
    end=10,
    step=1,
    value=1,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property1_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'Average Voltage (V)' property in the <b><i>ranking function</i></b>."
)
weights["Average Voltage (V)"] = w_property1
weights_helper["Average Voltage (V)"] = w_property1_help
# Property 2
w_property2 = pn.widgets.IntSlider(
    name="Average Voltage (deviation) (V)",
    start=-10,
    end=10,
    step=1,
    value=1,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property2_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'Average Voltage (deviation) (V)' property in the <b><i>ranking function</i></b>."
)
weights["Average Voltage (deviation) (V)"] = w_property2
weights_helper["Average Voltage (deviation) (V)"] = w_property2_help
# Property 3
w_property3 = pn.widgets.IntSlider(
    name="AI-experts confidence (-)",
    start=-10,
    end=10,
    step=1,
    value=1,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property3_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'AI-experts confidence (-)' property in the <b><i>ranking function</i></b>."
)
weights["AI-experts confidence (-)"] = w_property3
weights_helper["AI-experts confidence (-)"] = w_property3_help
# Property 4
w_property4 = pn.widgets.IntSlider(
    name="AI-experts confidence (deviation) (-)",
    start=-10,
    end=10,
    step=1,
    value=1,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property4_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'AI-experts confidence (deviation) (-)' property in the <b><i>ranking function</i></b>."
)
weights["AI-experts confidence (deviation) (-)"] = w_property4
weights_helper["AI-experts confidence (deviation) (-)"] = w_property4_help
# Property 5
w_property5 = pn.widgets.IntSlider(
    name="Max Volume expansion (%)",
    start=-10,
    end=10,
    step=1,
    value=1,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property5_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'Max Volume expansion (%)' property in the <b><i>ranking function</i></b>."
)
weights["Max Volume expansion (%)"] = w_property5
weights_helper["Max Volume expansion (%)"] = w_property5_help
# Property 6
w_property6 = pn.widgets.IntSlider(
    name="Max Volume expansion (deviation) (%)",
    start=-10,
    end=10,
    step=1,
    value=1,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property6_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'Max Volume expansion (deviation) (%)' property in the <b><i>ranking function</i></b>."
)
weights["Max Volume expansion (deviation) (%)"] = w_property6
weights_helper["Max Volume expansion (deviation) (%)"] = w_property6_help
# Property 7
w_property7 = pn.widgets.IntSlider(
    name="Stability charge (eV/atom)",
    start=-10,
    end=10,
    step=1,
    value=1,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property7_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'Stability charge (eV/atom)' property in the <b><i>ranking function</i></b>."
)
weights["Stability charge (eV/atom)"] = w_property7
weights_helper["Stability charge (eV/atom)"] = w_property7_help
# Property 8
w_property8 = pn.widgets.IntSlider(
    name="Stability charge (deviation) (eV/atom)",
    start=-10,
    end=10,
    step=1,
    value=1,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property8_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'Stability charge (deviation) (eV/atom)' property in the <b><i>ranking function</i></b>."
)
weights["Stability charge (deviation) (eV/atom)"] = w_property8
weights_helper["Stability charge (deviation) (eV/atom)"] = w_property8_help
# Property 9
w_property9 = pn.widgets.IntSlider(
    name="Stability discharge (eV/atom)",
    start=-10,
    end=10,
    step=1,
    value=1,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property9_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'Stability discharge (eV/atom)' property in the <b><i>ranking function</i></b>."
)
weights["Stability discharge (eV/atom)"] = w_property9
weights_helper["Stability discharge (eV/atom)"] = w_property9_help
# Property 10
w_property10 = pn.widgets.IntSlider(
    name="Stability discharge (deviation) (eV/atom)",
    start=-10,
    end=10,
    step=1,
    value=1,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property10_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'Stability discharge (deviation) (eV/atom)' property in the <b><i>ranking function</i></b>."
)
weights["Stability discharge (deviation) (eV/atom)"] = w_property10
weights_helper["Stability discharge (deviation) (eV/atom)"] = w_property10_help
# Property 11
w_property11 = pn.widgets.IntSlider(
    name="Volumetric capacity (mAh/L)",
    start=-10,
    end=10,
    step=1,
    value=1,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property11_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'Volumetric capacity (mAh/L)' property in the <b><i>ranking function</i></b>."
)
weights["Volumetric capacity (mAh/L)"] = w_property11
weights_helper["Volumetric capacity (mAh/L)"] = w_property11_help
# Property 12
w_property12 = pn.widgets.IntSlider(
    name="Gravimetric capacity (mAh/g)",
    start=-10,
    end=10,
    step=1,
    value=1,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property12_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'Gravimetric capacity (mAh/g)' property in the <b><i>ranking function</i></b>."
)
weights["Gravimetric capacity (mAh/g)"] = w_property12
weights_helper["Gravimetric capacity (mAh/g)"] = w_property12_help
# Property 13
w_property13 = pn.widgets.IntSlider(
    name="Volumetric energy (Wh/L)",
    start=-10,
    end=10,
    step=1,
    value=1,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property13_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'Volumetric energy (Wh/L)' property in the <b><i>ranking function</i></b>."
)
weights["Volumetric energy (Wh/L)"] = w_property13
weights_helper["Volumetric energy (Wh/L)"] = w_property13_help
# Property 14
w_property14 = pn.widgets.IntSlider(
    name="Gravimetric energy (Wh/kg)",
    start=-10,
    end=10,
    step=1,
    value=1,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property14_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'Gravimetric energy (Wh/kg)' property in the <b><i>ranking function</i></b>."
)
weights["Gravimetric energy (Wh/kg)"] = w_property14
weights_helper["Gravimetric energy (Wh/kg)"] = w_property14_help

# (2) Widget SIDEBAR : properties range
sliders = {}
sliders_helper = {}
# Property 1
s_property1 = create_range_slider("Average Voltage (V)", "Average Voltage (V)")
s_property1_help = pn.widgets.TooltipIcon(value="<b>Average Voltage (V)</b> Average voltage predicted by the ensemble committee of four E3NN models.")
sliders["Average Voltage (V)"] = s_property1
sliders_helper["Average Voltage (V)"] = s_property1_help
# Property 2
s_property2 = create_range_slider(
    "AI-experts confidence (-)", "AI-experts confidence (-)"
)
s_property2_help = pn.widgets.TooltipIcon(value="<b>AI-experts confidence (-)</b> Confidence level of the ensemble committee of ten GBDT models in classifying the material as a cathode.")
sliders["AI-experts confidence (-)"] = s_property2
sliders_helper["AI-experts confidence (-)"] = s_property2_help
# Property 3
s_property3 = create_range_slider("Max Volume expansion (%)", "Property 3 (-)")
s_property3_help = pn.widgets.TooltipIcon(value="<b>Property 3 (-)</b> description...")
sliders["Max Volume expansion (%)"] = s_property3
sliders_helper["Max Volume expansion (%)"] = s_property3_help
# Property 4
s_property4 = create_range_slider(
    "Stability charge (eV/atom)", "Property 4 (-)"
)
s_property4_help = pn.widgets.TooltipIcon(value="<b>Property 4 (-)</b> description...")
sliders["Stability charge (eV/atom)"] = s_property4
sliders_helper["Stability charge (eV/atom)"] = s_property4_help
# Property 5
s_property5 = create_range_slider(
    "Stability discharge (eV/atom)", "Property 5 (-)"
)
s_property5_help = pn.widgets.TooltipIcon(value="<b>Property 5 (-)</b> description...")
sliders["Stability discharge (eV/atom)"] = s_property5
sliders_helper["Stability discharge (eV/atom)"] = s_property5_help
# Property 6
s_property6 = create_range_slider("Volumetric capacity (mAh/L)", "Property 6 (-)")
s_property6_help = pn.widgets.TooltipIcon(value="<b>Property 6 (-)</b> description...")
sliders["Volumetric capacity (mAh/L)"] = s_property6
sliders_helper["Volumetric capacity (mAh/L)"] = s_property6_help
# Property 7
s_property7 = create_range_slider("Gravimetric capacity (mAh/g)", "Property 7 (-)")
s_property7_help = pn.widgets.TooltipIcon(value="<b>Property 7 (-)</b> description...")
sliders["Gravimetric capacity (mAh/g)"] = s_property7
sliders_helper["Gravimetric capacity (mAh/g)"] = s_property7_help
# Property 8
s_property8 = create_range_slider("Volumetric energy (Wh/L)", "Property 8 (-)")
s_property8_help = pn.widgets.TooltipIcon(value="<b>Property 8 (-)</b> description...")
sliders["Volumetric energy (Wh/L)"] = s_property8
sliders_helper["Volumetric energy (Wh/L)"] = s_property8_help

# (3) Widget SIDEBAR: Ions selection
select_ions = pn.widgets.MultiChoice(
    value=WORKING_IONS_ACTIVE,
    options=WORKING_IONS,
    #  sizing_mode='stretch_width',
    width=SIDEBAR_WIDGET_W,
    sizing_mode="fixed",
    description="Add or remove <i>cathodes</i> with a specific <i>active ion material</i>",
)


# (1) Widget MAIN: Table properties
select_properties = pn.widgets.MultiChoice(
    name="Database Properties",
    value=COLUMNS_ACTIVE,
    options=COLUMNS,
    sizing_mode="stretch_width",
)

# (2) Widget MAIN: Table
editors = {key: {"disabled": True} for key in df}
downloadable_table = pn.bind(
    build_interactive_table,
    # weights
    w_property1=w_property1,
    w_property2=w_property2,
    w_property3=w_property3,
    w_property4=w_property4,
    w_property5=w_property5,
    w_property6=w_property6,
    w_property7=w_property7,
    w_property8=w_property8,
    w_property9=w_property9,
    w_property10=w_property10,
    w_property11=w_property11,
    w_property12=w_property12,
    w_property13=w_property13,
    w_property14=w_property14,
    # sliders
    # s_classifier_mean=s_classifier_mean,
    columns=select_properties,
    categories=select_ions,
    sliders=sliders,
)

# (3) Widget MAIN: Plot
plot = pn.bind(
    build_interactive_plot,
    s_property1=s_property1,
    s_property2=s_property2,
    s_property3=s_property3,
    s_property4=s_property4,
    s_property5=s_property5,
    s_property6=s_property6,
    s_property7=s_property7,
    s_property8=s_property8,
    categories=select_ions,
)

# Widget MAIN: Text
text_info = pn.pane.Markdown(ABOUT_MSG, width=ABOUT_W)
download_bibtex = pn.widgets.FileDownload(
    icon="download",
    label="Download BibTeX ",
    button_type="primary",
    filename="reference.bib",
    callback=partial(get_raw_file_github, BIB_FILE),
    embed=True,
)
download_ris = pn.widgets.FileDownload(
    icon="download",
    label="Download RIS ",
    button_type="primary",
    filename="reference.ris",
    callback=partial(get_raw_file_github, RIS_FILE),
    embed=True,
)

about_box = pn.Column(text_info, pn.Row(download_bibtex, download_ris))

# Layout
weights_col = pn.Column()
for key in weights.keys():
    weights_col.append(pn.Row(weights[key], weights_helper[key]))

sliders_col = pn.Column()
for key in sliders.keys():
    sliders_col.append(pn.Row(sliders[key], sliders_helper[key]))

controls_tabs_intro = pn.pane.Markdown(
    """
<style>
p, h1, h2, h3 {
    margin-block-start: 0.2em;
    margin-block-end: 0.2em;
}
ul {
    margin-block-start: 0.3em;
    margin-block-end: 0.3em;
}
</style>
## Control panel
The control panel below has two tabs:
* **Properties**: Allows you to filter the database by controlling the ranges of different properties.
* **Ranking**: Allows you to control the ranking score by adjusting the weights applied to different min-max normalized properties.""",
    # styles={'margin-block-start': "0.5em"},
)

controls_tabs = pn.Tabs(("Properties", sliders_col), ("Ranking", weights_col))

box_select_ions = pn.Column(
    pn.Row(
        pn.pane.Markdown(
            """
<style>
p, h1, h2, h3 {
    margin-block-start: 0.2em;
    margin-block-end: 0.2em;
}
ul {
    margin-block-start: 0.3em;
    margin-block-end: 0.3em;
}
</style>
## Working categories
Add or remove rows belloging to specific category"""
        ),
        #    pn.widgets.TooltipIcon(
        #        value="Add or remove <i>cathodes</i> with a specific <i>active ion material</i>"
        #        )
    ),
    select_ions,
)

divider_sb = pn.layout.Divider(margin=(-5, 0, -5, 0))
divider_m = pn.layout.Divider()
footer = pn.pane.HTML(FOOTER, sizing_mode="stretch_width")


pn.template.FastListTemplate(
    title=TITLE,
    sidebar=[box_select_ions, divider_sb, controls_tabs_intro, controls_tabs],
    main=[
        pn.Row(plot, about_box),
        pn.Column(select_properties, downloadable_table),
        divider_m,
        footer,
    ],
    sidebar_width=SIDEBAR_W,
    # header_background = "CadetBlue",
    main_layout=None,
    accent=ACCENT,
).servable()
