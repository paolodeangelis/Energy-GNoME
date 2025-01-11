from functools import partial
from io import StringIO
from itertools import product

from bokeh.models import HTMLTemplateFormatter
import hvplot.pandas
import numpy as np
import pandas as pd
import panel as pn
import requests

pn.extension("tabulator")
pn.extension(throttled=True)

# CONSTANTS (settings)
SITE = "Energy-GNoME"
SITE_URL = "https://paolodeangelis.github.io/Energy-GNoME/apps/"
FAVICON = "https://raw.githubusercontent.com/paolodeangelis/Energy-GNoME/main/docs/assets/img/favicon.png"
TITLE = "Cathode materials explorer"
LOGO = "https://raw.githubusercontent.com/paolodeangelis/Energy-GNoME/main/docs/assets/img/logo_alt.png"
DATA_PATH_TEMPLATE = "https://raw.githubusercontent.com/paolodeangelis/Energy-GNoME/main/data/final/cathodes/{ctype}/{ion}/candidates.json"
BIB_FILE = "https://raw.githubusercontent.com/paolodeangelis/Energy-GNoME/main/assets/cite/energy-gnome.bib"
RIS_FILE = "https://raw.githubusercontent.com/paolodeangelis/Energy-GNoME/main/assets/cite/energy-gnome.ris"
RTF_FILE = "https://raw.githubusercontent.com/paolodeangelis/Energy-GNoME/main/assets/cite/energy-gnome.rtf"
ARTICLE_DOI = "10.48550/arXiv.2411.10125"
ARTICLE_TEXT_CITE = f'De Angelis, P.; Trezza, G.; Barletta, G.; Asinari, P.; Chiavazzo, E. "Energy-GNoME: A Living Database of Selected Materials for Energy Applications". *arXiv* November 15, **2024**. doi: <a href="https://doi.org/{ARTICLE_DOI}" target="_blank">{ARTICLE_DOI}</a>.'
DOC_PAGE = "https://paolodeangelis.github.io/Energy-GNoME/apps/cathodes/userguide/"
ACCENT = "#4fc4d3"
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
FONT = {
    "name": "Roboto",
    "url": "https://fonts.googleapis.com/css2?family=Noto+Sans+Math&family=Roboto",
}
WORKING_IONS = ["Li", "Na", "Mg", "K", "Ca", "Cs", "Al", "Rb", "Y"]
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
    "Max Volume expansion (-)",
    "Max Volume expansion (deviation) (-)",
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
    "Note",
]
HOVER_COL = [
    ("Material Id", "@{Material Id}"),
    ("Working Ion", "@{Working Ion}"),
    ("Average Voltage", "@{Average Voltage (V)}{0.2f} V"),
    ("AI-experts confidence", "@{AI-experts confidence (-)}{0.2f}"),
    ("Max Volume expansion", "@{Max Volume expansion (-)}{0.2f} L/L"),
    ("Volumetric capacity", "@{Volumetric capacity (mAh/L)}{0.2f} mAh/L"),
    ("Gravimetric capacity", "@{Gravimetric capacity (mAh/g)}{0.2f} mAh/g"),
    ("Volumetric energy", "@{Volumetric energy (Wh/L)}{0.2f} Wh/L"),
    ("Gravimetric energy", "@{Gravimetric energy (Wh/kg)}{0.2f} Wh/kg"),
]

COLUMNS_ACTIVE = [
    "Material Id",
    "Formula",
    # "Volumetric capacity (mAh/L)",
    "Gravimetric capacity (mAh/g)",
    # "Volumetric energy (Wh/L)",
    "Gravimetric energy (Wh/kg)",
    "Average Voltage (V)",
    "AI-experts confidence (-)",
    "Ranking",
    "File",
    "Note",
]
N_ROW = 12
SIDEBAR_W = 350
SIDEBAR_WIDGET_W = 290
PLOT_SIZE = [850, 550]  # WxH
TABLE_FORMATTER = {
    # "File": HTMLTemplateFormatter(template=r'<code><a href="https://raw.githubusercontent.com/paolodeangelis/Energy-GNoME/main/<%= _folder_path %>/<%= value %>.CIF?download=1" download="<%= value %>.CIF" rel="noopener noreferrer" target="_blank"> <i class="fas fa-external-link-alt"></i> <%= value %>.CIF </a></code>') # Problem with RawGithub link (it open it as txt file) # noqa:W505
    # "File": HTMLTemplateFormatter(
    #     template=r'<code><a href="https://github.com/paolodeangelis/Energy-GNoME/blob/main/<%= _folder_path %>/<%= value %>.CIF?download=1" download="<%= value %>.CIF" rel="noopener noreferrer" target="_blank"> <i class="fas fa-external-link-alt"></i> <%= value %>.CIF </a></code>' # noqa:W505
    # )
    "File": HTMLTemplateFormatter(
        template=r'<code><a href="https://paolodeangelis.github.io/Energy-GNoME/materials/<%= value %>" target="_blank"> <i class="fas fa-external-link-alt"></i> <%= value %>.CIF </a></code>'
    ),
    "Material Id": HTMLTemplateFormatter(
        template=r'<code><a href="https://paolodeangelis.github.io/Energy-GNoME/materials/<%= value %>" target="_blank"> <i class="fas fa-external-link-alt"></i> <%= value %> </a></code>'
    ),
}
ABOUT_W = 600
ABOUT_MSG = f"""
# Usage

This dashboard allows you to explore candidate cathode materials from the GNoME database.

On the left sidebar, you can dynamically filter the materials displayed on the scatter plot and in the table below. Use the sliders to set thresholds for various properties, which act as filters to narrow down the database to the most relevant materials for your needs.

The ranking function enables you to prioritize materials based on your criteria. You can adjust the weights for each property directly in the widget bar to influence the ranking score.

Once you've refined your search and explored the materials, you can download the filtered list as a .CSV file for more detailed analysis. Additionally, you can use the links in the results table to download the corresponding CIF files.

For in-depth guidance or further details about the features, please refer to the [documentation pages]({DOC_PAGE}).

If you find this dataset valuable, please consider citing the original work:

> {ARTICLE_TEXT_CITE}

"""
META = {
    "description": "Explore advanced cathode material analysis and Artificial Intelligence screening with interactive tools from the GNoME database.",
    "keywords": "cathode materials, GNoME database, material analysis, battery research, interactive dashboard, artificial intelligence",
    "authors": "Paolo De Angelis, Giulio Barletta, Giovanni Trezza",
    "viewport": f"width={SIDEBAR_W + SIDEBAR_WIDGET_W + PLOT_SIZE[0] + ABOUT_W:d}px, initial-scale=1",
}
FOOTER = r"""
<link rel="stylesheet"
href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/dist/tabler-icons.min.css" />
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.1/css/all.min.css">


<style>
    .app-footer-meta {
        display: flex; /* Use Flexbox layout */
        justify-content: space-between; /* Align items to left, center, and right */
        align-items: center; /* Center items vertically */
        gap: 1rem; /* Add space between items */
        flex-wrap: wrap; /* Allow wrapping for smaller screens */
    }

    .app-copyright {
        flex: 0 0 auto; /* Fixed size, do not grow or shrink */
        min-width: 120px; /* Ensure enough space */
        text-align: left; /* Align text to the left */
    }

    .app-footer-madeby {
        flex: 1 1 auto; /* Flexible: takes remaining space */
        text-align: center; /* Center text within this section */
    }

    .app-social {
        flex: 0 0 auto; /* Fixed size, do not grow or shrink */
        min-width: 120px; /* Ensure enough space */
        text-align: right; /* Align text to the right */
        display: inline-flex; /* Align social icons horizontally */
        gap: 0.5rem; /* Add space between icons */
    }

</style>

<footer class="app-footer" style="color: #5a5a5a; text-decoration: none;">
    <div class="app-footer-meta">
        <div class="app-copyright">
            Copyright &copy; 2025 <a href="https://areeweb.polito.it/ricerca/small/" target="_blank" style="color: #5a5a5a; text-decoration: none;" onmouseover="this.style.color='#929292'" onmouseout="this.style.color='#5a5a5a'"> Small Lab </a>
         </div>

        <div class="app-footer-madeby" >
            <p>
                <i class="ti ti-code"
                style="vertical-align: middle; font-size: 1.25em;"></i>
                with
                <i class="ti ti-heart-filled"
                style="vertical-align: middle; font-size: 1.25em;"></i>
                by
                <a href="https://paolodeangelis.github.io/" target="_blank" style="color: #5a5a5a; text-decoration: none;" onmouseover="this.style.color='#929292'" onmouseout="this.style.color='#5a5a5a'">
                    <strong >
                    PDA
                    </strong>
                </a>
                &
                <a href="https://giuliobarl.github.io/" target="_blank" style="color: #5a5a5a; text-decoration: none;" onmouseover="this.style.color='#929292'" onmouseout="this.style.color='#5a5a5a'">
                    <strong >
                    GB
                    </strong>
                </a>
                <br>
                    Made entirely with OSS packages:
                    <a href="https://panel.holoviz.org/" target="_blank"target="_blank" style="color: #5a5a5a; text-decoration: none;" onmouseover="this.style.color='#929292'" onmouseout="this.style.color='#5a5a5a'">
                    <strong>Panel</strong>
                    </a>,
                    <a href="https://holoviews.org/" target="_blank" style="color: #5a5a5a; text-decoration: none;" onmouseover="this.style.color='#929292'" onmouseout="this.style.color='#5a5a5a'">
                    <strong>Holoviews</strong>
                    </a>,
                    <a href="https://bokeh.org/" target="_blank" style="color: #5a5a5a; text-decoration: none;" onmouseover="this.style.color='#929292'" onmouseout="this.style.color='#5a5a5a'">
                    <strong>Bokeh</strong>
                    </a>,
                    <a href="https://pandas.pydata.org/" target="_blank" style="color: #5a5a5a; text-decoration: none;" onmouseover="this.style.color='#929292'" onmouseout="this.style.color='#5a5a5a'">
                    <strong>Pandas</strong>
                    </a>,
                    <a href="https://numpy.org/" target="_blank" style="color: #5a5a5a; text-decoration: none;" onmouseover="this.style.color='#929292'" onmouseout="this.style.color='#5a5a5a'">
                    <strong>Numpy</strong>
                    </a>.
                </p>
            </div>
            <div class="app-social">
                <a href="https://github.com/paolodeangelis/Energy-GNoME/" target="_blank" style="color: #5a5a5a; text-decoration: none;" onmouseover="this.style.color='#929292'" onmouseout="this.style.color='#5a5a5a'">
                    <i class="fa-brands fa-github"></i>
                </a>
                <a href="https://x.com/small_polito" target="_blank" style="color: #5a5a5a; text-decoration: none;" onmouseover="this.style.color='#929292'" onmouseout="this.style.color='#5a5a5a'">
                    <i class="fa-brands fa-x-twitter"></i>
                </a>
                <a href="https://www.linkedin.com/company/small-lab/" target="_blank" style="color: #5a5a5a; text-decoration: none;" onmouseover="this.style.color='#929292'" onmouseout="this.style.color='#5a5a5a'">
                    <i class="fa-brands fa-linkedin"></i>
                </a>
             </div>
        </div>
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
        df["_folder_path"] = f"data/final/cathodes/{ctype}/{ion}/cif"
        df.insert(len(df.columns) - 1, "Note", df.pop("Note"))
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


def down_load_menu(filename, table):
    """
    Create a download button for the filtered table data.

    Args:
        filename (str): The name of the file to download.
        table (pn.widgets.Tabulator): The table widget to get the filtered data from.

    Returns:
        pn.widgets.FileDownload: A file download widget with the filtered table data.
    """

    def create_file():
        sio = StringIO()
        filtered_data = table.current_view  # Get the current filtered view of the table
        filtered_data.to_csv(sio, index=False)
        sio.seek(0)  # Rewind the StringIO buffer to the beginning
        return sio  # Return the StringIO object (Panel handles it properly for download)

    button = pn.widgets.FileDownload(
        callback=create_file, label="Download filtered database", filename=filename
    )
    return button


def build_interactive_table(
    # weights
    w_property1: pn.widgets.FloatSlider,
    w_property2: pn.widgets.FloatSlider,
    w_property3: pn.widgets.FloatSlider,
    w_property4: pn.widgets.FloatSlider,
    w_property5: pn.widgets.FloatSlider,
    w_property6: pn.widgets.FloatSlider,
    w_property7: pn.widgets.FloatSlider,
    w_property8: pn.widgets.FloatSlider,
    w_property9: pn.widgets.FloatSlider,
    w_property10: pn.widgets.FloatSlider,
    w_property11: pn.widgets.FloatSlider,
    w_property12: pn.widgets.FloatSlider,
    w_property13: pn.widgets.FloatSlider,
    w_property14: pn.widgets.FloatSlider,
    # sliders
    # s_classifier_mean: pn.widgets.RangeSlider,
    columns: list,
    sliders: dict = None,
    categories: list = None,
) -> pn.Column:
    """
    Build an interactive table with ranking and filtering features based on the provided weights and filters.

    Args:
        w_property1 to w_property8: FloatSlider widgets representing weights for each property.
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
        + w_property5 * min_max_norm(df["Max Volume expansion (-)"])
        + w_property6 * min_max_norm(df["Max Volume expansion (deviation) (-)"])
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
                table.add_filter(
                    pn.bind(
                        apply_range_filter, column=column, value_range=slider.param.value_throttled
                    )
                )
    # Apply category filters for categories
    if categories:
        hidden_ions = set(all_ions) - set(categories)
        for ion in hidden_ions:
            table.add_filter(pn.bind(apply_category_filter, category=CATEGORY, item_to_hide=ion))

    # Watch sliders and update download button
    def update_download(event):
        # Recreate the download button whenever filters or sliders change
        button = down_load_menu(filename.param.value, table)
        return button

    for slider in sliders.values():
        slider.param.watch(update_download, "value_throttled")

    # Add download section
    filename = pn.widgets.TextInput(name="Enter filename", value="cathode_candidates.csv")
    button = down_load_menu(filename.param.value, table)

    # Function to update the download button's filename dynamically
    def update_filename(event):
        button.filename = event.new  # Update the filename of the FileDownload button

    # Watch the filename input and update the download button filename
    filename.param.watch(update_filename, "value")
    layout = pn.Column(filename, button, table)

    return layout


def build_interactive_plot(
    s_property1: pn.widgets.RangeSlider,
    s_property2: pn.widgets.RangeSlider,
    s_property3: pn.widgets.RangeSlider,
    s_property4: pn.widgets.RangeSlider,
    s_property5: pn.widgets.RangeSlider,
    s_property6: pn.widgets.RangeSlider,
    s_property7: pn.widgets.RangeSlider,
    s_property8: pn.widgets.RangeSlider,
    s_property9: pn.widgets.RangeSlider,
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
        ("Max Volume expansion (-)", s_property3),
        ("Stability charge (eV/atom)", s_property4),
        ("Stability discharge (eV/atom)", s_property5),
        ("Volumetric capacity (mAh/L)", s_property6),
        ("Gravimetric capacity (mAh/g)", s_property7),
        ("Volumetric energy (Wh/L)", s_property8),
        ("Gravimetric energy (Wh/kg)", s_property9),
    ]

    for col, slider in filters:
        plot_df = plot_df[(plot_df[col] >= slider[0]) & (plot_df[col] <= slider[1])]

    # Apply ion filter if provided
    if categories:
        plot_df = plot_df[plot_df[CATEGORY].isin(categories)]

    # Background scatter plot with all data
    back_scatter = df.hvplot.scatter(
        x="Gravimetric capacity (mAh/g)",
        y="Average Voltage (V)",
        s=100,
        alpha=0.25,
        color="#444",
        line_color="white",
        hover_cols="all",
    ).opts(
        tools=[],
        logx=True,
        logy=False,
        xlabel="Gravimetric capacity (mAh/g)",
        ylabel="Average Voltage (V)",
    )

    # Foreground scatter plot with filtered data
    front_scatter = plot_df.hvplot.scatter(
        x="Gravimetric capacity (mAh/g)",
        y="Average Voltage (V)",
        s=100,
        line_color="white",
        c=CATEGORY,
        legend="top",
        hover_cols="all",
    ).opts(
        logx=True,
        logy=False,
        xlabel="Gravimetric capacity (mAh/g)",
        ylabel="Average Voltage (V)",
        cmap=PALETTE,
        # tools=[hover],
        hover_tooltips=HOVER_COL,
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
w_property1 = pn.widgets.FloatSlider(
    name="Average Voltage (V)",
    start=-10,
    end=10,
    step=0.5,
    value=0,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property1_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'Average Voltage (V)' property in the <b><i>ranking function</i></b>."
)
weights["Average Voltage (V)"] = w_property1
weights_helper["Average Voltage (V)"] = w_property1_help
# Property 2
w_property2 = pn.widgets.FloatSlider(
    name="Average Voltage (deviation) (V)",
    start=-10,
    end=10,
    step=0.5,
    value=-2,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property2_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'Average Voltage (deviation) (V)' property in the <b><i>ranking function</i></b>."
)
weights["Average Voltage (deviation) (V)"] = w_property2
weights_helper["Average Voltage (deviation) (V)"] = w_property2_help
# Property 3
w_property3 = pn.widgets.FloatSlider(
    name="AI-experts confidence (-)",
    start=-10,
    end=10,
    step=0.5,
    value=4,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property3_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'AI-experts confidence (-)' property in the <b><i>ranking function</i></b>."
)
weights["AI-experts confidence (-)"] = w_property3
weights_helper["AI-experts confidence (-)"] = w_property3_help
# Property 4
w_property4 = pn.widgets.FloatSlider(
    name="AI-experts confidence (deviation) (-)",
    start=-10,
    end=10,
    step=0.5,
    value=-2,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property4_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'AI-experts confidence (deviation) (-)' property in the <b><i>ranking function</i></b>."
)
weights["AI-experts confidence (deviation) (-)"] = w_property4
weights_helper["AI-experts confidence (deviation) (-)"] = w_property4_help
# Property 5
w_property5 = pn.widgets.FloatSlider(
    name="Max Volume expansion (-)",
    start=-10,
    end=10,
    step=0.5,
    value=-0.5,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property5_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'Max Volume expansion (-)' property in the <b><i>ranking function</i></b>."
)
weights["Max Volume expansion (-)"] = w_property5
weights_helper["Max Volume expansion (-)"] = w_property5_help
# Property 6
w_property6 = pn.widgets.FloatSlider(
    name="Max Volume expansion (deviation) (-)",
    start=-10,
    end=10,
    step=0.5,
    value=0,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property6_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'Max Volume expansion (deviation) (-)' property in the <b><i>ranking function</i></b>."
)
weights["Max Volume expansion (deviation) (-)"] = w_property6
weights_helper["Max Volume expansion (deviation) (-)"] = w_property6_help
# Property 7
w_property7 = pn.widgets.FloatSlider(
    name="Stability charge (eV/atom)",
    start=-10,
    end=10,
    step=0.5,
    value=-0.5,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property7_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'Stability charge (eV/atom)' property in the <b><i>ranking function</i></b>."
)
weights["Stability charge (eV/atom)"] = w_property7
weights_helper["Stability charge (eV/atom)"] = w_property7_help
# Property 8
w_property8 = pn.widgets.FloatSlider(
    name="Stability charge (deviation) (eV/atom)",
    start=-10,
    end=10,
    step=0.5,
    value=0,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property8_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'Stability charge (deviation) (eV/atom)' property in the <b><i>ranking function</i></b>."
)
weights["Stability charge (deviation) (eV/atom)"] = w_property8
weights_helper["Stability charge (deviation) (eV/atom)"] = w_property8_help
# Property 9
w_property9 = pn.widgets.FloatSlider(
    name="Stability discharge (eV/atom)",
    start=-10,
    end=10,
    step=0.5,
    value=-0.5,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property9_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'Stability discharge (eV/atom)' property in the <b><i>ranking function</i></b>."
)
weights["Stability discharge (eV/atom)"] = w_property9
weights_helper["Stability discharge (eV/atom)"] = w_property9_help
# Property 10
w_property10 = pn.widgets.FloatSlider(
    name="Stability discharge (deviation) (eV/atom)",
    start=-10,
    end=10,
    step=0.5,
    value=0,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property10_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'Stability discharge (deviation) (eV/atom)' property in the <b><i>ranking function</i></b>."
)
weights["Stability discharge (deviation) (eV/atom)"] = w_property10
weights_helper["Stability discharge (deviation) (eV/atom)"] = w_property10_help
# Property 11
w_property11 = pn.widgets.FloatSlider(
    name="Volumetric capacity (mAh/L)",
    start=-10,
    end=10,
    step=0.5,
    value=0,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property11_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'Volumetric capacity (mAh/L)' property in the <b><i>ranking function</i></b>."
)
weights["Volumetric capacity (mAh/L)"] = w_property11
weights_helper["Volumetric capacity (mAh/L)"] = w_property11_help
# Property 12
w_property12 = pn.widgets.FloatSlider(
    name="Gravimetric capacity (mAh/g)",
    start=-10,
    end=10,
    step=0.5,
    value=2,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property12_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'Gravimetric capacity (mAh/g)' property in the <b><i>ranking function</i></b>."
)
weights["Gravimetric capacity (mAh/g)"] = w_property12
weights_helper["Gravimetric capacity (mAh/g)"] = w_property12_help
# Property 13
w_property13 = pn.widgets.FloatSlider(
    name="Volumetric energy (Wh/L)",
    start=-10,
    end=10,
    step=0.5,
    value=0,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property13_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'Volumetric energy (Wh/L)' property in the <b><i>ranking function</i></b>."
)
weights["Volumetric energy (Wh/L)"] = w_property13
weights_helper["Volumetric energy (Wh/L)"] = w_property13_help
# Property 14
w_property14 = pn.widgets.FloatSlider(
    name="Gravimetric energy (Wh/kg)",
    start=-10,
    end=10,
    step=0.5,
    value=2,
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
s_property1_help = pn.widgets.TooltipIcon(
    value="<b>Average Voltage (V)</b>: Average voltage predicted by the ensemble committee of four E3NN models."
)
sliders["Average Voltage (V)"] = s_property1
sliders_helper["Average Voltage (V)"] = s_property1_help
# Property 2
s_property2 = create_range_slider("AI-experts confidence (-)", "AI-experts confidence (-)")
s_property2_help = pn.widgets.TooltipIcon(
    value="<b>AI-experts confidence (-)</b>: Confidence level of the ensemble committee of ten GBDT models in classifying the material as a cathode."
)
sliders["AI-experts confidence (-)"] = s_property2
sliders_helper["AI-experts confidence (-)"] = s_property2_help
# Property 3
s_property3 = create_range_slider("Max Volume expansion (-)", "Max Volume expansion (-)")
s_property3_help = pn.widgets.TooltipIcon(
    value="<b>Max Volume expansion (-)</b>: Predicted maximum volume expansion (<i>V<sub>max</sub>/V<sub>min</sub></i>)of the cathode during discharge, estimated by the ensemble committee of four E3NN models."
)
sliders["Max Volume expansion (-)"] = s_property3
sliders_helper["Max Volume expansion (-)"] = s_property3_help
# Property 4
s_property4 = create_range_slider("Stability charge (eV/atom)", "Stability charge (eV/atom)")
s_property4_help = pn.widgets.TooltipIcon(
    value="<b>Stability charge (eV/atom)</b>: Predicted energy above the hull for the specified charge state."
)
sliders["Stability charge (eV/atom)"] = s_property4
sliders_helper["Stability charge (eV/atom)"] = s_property4_help
# Property 5
s_property5 = create_range_slider("Stability discharge (eV/atom)", "Stability discharge (eV/atom)")
s_property5_help = pn.widgets.TooltipIcon(
    value="<b>Stability discharge (eV/atom)</b>: Predicted energy above the hull for the specified discharge state."
)
sliders["Stability discharge (eV/atom)"] = s_property5
sliders_helper["Stability discharge (eV/atom)"] = s_property5_help
# Property 6
s_property6 = create_range_slider("Volumetric capacity (mAh/L)", "Volumetric capacity (mAh/L)")
s_property6_help = pn.widgets.TooltipIcon(
    value="<b>Volumetric capacity (mAh/L)</b>:  Capacity denisty of the pure cathode material."
)
sliders["Volumetric capacity (mAh/L)"] = s_property6
sliders_helper["Volumetric capacity (mAh/L)"] = s_property6_help
# Property 7
s_property7 = create_range_slider("Gravimetric capacity (mAh/g)", "Gravimetric capacity (mAh/g)")
s_property7_help = pn.widgets.TooltipIcon(
    value="<b>Gravimetric capacity (mAh/g)</b>: Specific capacity of the pure cathode material."
)
sliders["Gravimetric capacity (mAh/g)"] = s_property7
sliders_helper["Gravimetric capacity (mAh/g)"] = s_property7_help
# Property 8
s_property8 = create_range_slider("Volumetric energy (Wh/L)", "Volumetric energy (Wh/L)")
s_property8_help = pn.widgets.TooltipIcon(
    value="<b>Volumetric energy (Wh/L)</b>: Energy denisty of the pure cathode material."
)
sliders["Volumetric energy (Wh/L)"] = s_property8
sliders_helper["Volumetric energy (Wh/L)"] = s_property8_help
# Property 9
s_property9 = create_range_slider("Gravimetric energy (Wh/kg)", "Gravimetric energy (Wh/kg)")
s_property9_help = pn.widgets.TooltipIcon(
    value="<b>Gravimetric energy (Wh/kg)</b>: Specific energy of the pure cathode material."
)
sliders["Gravimetric energy (Wh/kg)"] = s_property9
sliders_helper["Gravimetric energy (Wh/kg)"] = s_property9_help

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
    s_property9=s_property9,
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
download_rtf = pn.widgets.FileDownload(
    icon="download",
    label="Download RTF ",
    button_type="primary",
    filename="reference.rtf",
    callback=partial(get_raw_file_github, RTF_FILE),
    embed=True,
)

about_box = pn.Column(
    text_info, pn.Row(download_bibtex, download_ris, download_rtf, styles=dict(margin="auto"))
)

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
## Working Ion
Easily manage cathodes associated with a specific working ion."""
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
    site=SITE,
    site_url=SITE_URL,
    favicon=FAVICON,
    title=TITLE,
    logo=LOGO,
    font=FONT["name"],
    font_url=FONT["url"],
    meta_author=META["authors"],
    meta_viewport=META["viewport"],
    meta_keywords=META["keywords"],
    meta_description=META["description"],
    theme_toggle=False,  # not working once static
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
