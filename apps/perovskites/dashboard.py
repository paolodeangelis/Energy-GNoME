from functools import partial
from io import StringIO

from bokeh.models import HoverTool, HTMLTemplateFormatter
import hvplot.pandas
import numpy as np
import pandas as pd
import panel as pn
import param
import requests

pn.extension("tabulator")
pn.extension(throttled=True)

# CONSTANTS (settings)
SITE = "Energy-GNoME"
SITE_URL = "https://paolodeangelis.github.io/Energy-GNoME/apps/"
FAVICON = "https://raw.githubusercontent.com/paolodeangelis/Energy-GNoME/main/docs/assets/img/favicon.png"
TITLE = "Perovskite materials explorer"
LOGO = "https://raw.githubusercontent.com/paolodeangelis/Energy-GNoME/main/docs/assets/img/logo_alt.png"
DATA_PATH = "https://raw.githubusercontent.com/paolodeangelis/Energy-GNoME/main/data/final/perovskites/{modeltype}/candidates.json"
BIB_FILE = "https://raw.githubusercontent.com/paolodeangelis/Energy-GNoME/main/assets/cite/energy-gnome.bib"
RIS_FILE = "https://raw.githubusercontent.com/paolodeangelis/Energy-GNoME/main/assets/cite/energy-gnome.ris"
RTF_FILE = "https://raw.githubusercontent.com/paolodeangelis/Energy-GNoME/main/assets/cite/energy-gnome.rtf"
ARTICLE_DOI = "10.48550/arXiv.2411.10125"
ARTICLE_TEXT_CITE = f'De Angelis, P.; Trezza, G.; Barletta, G.; Asinari, P.; Chiavazzo, E. "Energy-GNoME: A Living Database of Selected Materials for Energy Applications". *arXiv* November 15, **2024**. doi: <a href="https://doi.org/{ARTICLE_DOI}" target="_blank">{ARTICLE_DOI}</a>.'
DOC_PAGE = "https://paolodeangelis.github.io/Energy-GNoME/apps/perovskites/userguide/"
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
MODEL_TYPE = ["Pure Models", "Mixed Models"]
MODEL_ACTIVE = ["Pure Models"]
CATEGORY = "Model type"
CATEGORY_ACTIVE = MODEL_ACTIVE
ANGSTROM_SYMBOL = "\u212B"
CUBE_SYMBOL = "\u00B3"
COLUMNS = [
    "Material Id",
    "Composition",
    "Crystal System",
    "Formation Energy (eV/atom)",
    "Formula",
    f"Volume ({ANGSTROM_SYMBOL}{CUBE_SYMBOL})",
    f"Density ({ANGSTROM_SYMBOL}{CUBE_SYMBOL}/atom)",
    "Model type",
    "Average Band Gap (eV)",
    "Average Band Gap (deviation) (eV)",
    "AI-experts confidence (-)",
    "AI-experts confidence (deviation) (-)",
    "Ranking",
    "File",
]
HOVER_COL = [
    ("Material Id", "@{Material Id}"),
    ("Formula", "@{Formula}"),
    ("Formation Energy", "@{Formation Energy (eV/atom)}{0.2f} eV/atom"),
    ("Average Band Gap", "@{Average Band Gap (eV)}{0.2f} eV"),
    ("AI-experts confidence", "@{AI-experts confidence (-)}{0.2f}"),
]
COLUMNS_ACTIVE = [
    "Material Id",
    "Formula",
    "Average Band Gap (eV)",
    "AI-experts confidence (-)",
    "Ranking",
    "File",
]
N_ROW = 12
SIDEBAR_W = 350
SIDEBAR_WIDGET_W = 290
PLOT_SIZE = [850, 550]  # WxH
TABLE_FORMATTER = {
    # "File": HTMLTemplateFormatter(template=r'<code><a href="https://raw.githubusercontent.com/paolodeangelis/Energy-GNoME/main/<%= _folder_path %>/<%= value %>.CIF?download=1" download="<%= value %>.CIF" rel="noopener noreferrer" target="_blank"> <i class="fas fa-external-link-alt"></i> <%= value %>.CIF </a></code>') # Problem with RawGithub link (it open it as txt file) # noqa:W505
    "File": HTMLTemplateFormatter(
        template=r'<code><a href="https://github.com/paolodeangelis/Energy-GNoME/blob/main/<%= _folder_path %>/<%= value %>.CIF?download=1" download="<%= value %>.CIF" rel="noopener noreferrer" target="_blank"> <i class="fas fa-external-link-alt"></i> <%= value %>.CIF </a></code>'
    )
}
ABOUT_W = 600
ABOUT_MSG = f"""
# Usage

This dashboard allows you to explore candidate perovskite materials from the GNoME database.

On the left sidebar, you can dynamically filter the materials displayed on the scatter plot and in the table below. Use the sliders to set thresholds for various properties, which act as filters to narrow down the database to the most relevant materials for your needs.

The ranking function enables you to prioritize materials based on your criteria. You can adjust the weights for each property directly in the widget bar to influence the ranking score.

Once you've refined your search and explored the materials, you can download the filtered list as a .CSV file for more detailed analysis. Additionally, you can use the links in the results table to download the corresponding CIF files.

For in-depth guidance or further details about the features, please refer to the [documentation pages]({DOC_PAGE}).

If you find this dataset valuable, please consider citing the original work:

> {ARTICLE_TEXT_CITE}

"""
META = {
    "description": "Explore advanced perovskite material analysis and Artificial Intelligence screening with interactive tools from the GNoME database.",
    "keywords": "perovskite materials, GNoME database, material analysis, battery research, interactive dashboard, artificial intelligence",
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
            Copyright &copy; 2024 <a href="https://areeweb.polito.it/ricerca/small/" target="_blank" style="color: #5a5a5a; text-decoration: none;" onmouseover="this.style.color='#929292'" onmouseout="this.style.color='#5a5a5a'"> Small Lab </a>
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
    Load and initialize the datasets for different regression models by setting up default columns and values.

    This function loads the dataset from a JSON file, initializes a 'Ranking' column
    with default values of 1, and creates a 'File' column based on the 'ID' column.

    Returns:
        pd.DataFrame: The initialized DataFrame with additional columns.
    """

    # Use a generator to load and process data lazily
    def load_and_process(modeltype):
        _modeltype = "_".join(modeltype.split()).lower()
        path = DATA_PATH.format(modeltype=_modeltype)
        df = pd.read_json(path)
        df = df.rename(
            columns={
                "Volume (A^3)": f"Volume ({ANGSTROM_SYMBOL}{CUBE_SYMBOL})",
                "Density (A^3/atom)": f"Density ({ANGSTROM_SYMBOL}{CUBE_SYMBOL}/atom)",
            }
        )
        df["Model type"] = modeltype  # " ".join([s.capitalize() for s in modeltype.split("_")])
        df["Ranking"] = 1.0
        df["File"] = df["Material Id"]
        df["_folder_path"] = f"data/final/perovskites/{_modeltype}/cif"
        # Downcast float64 to float32 for memory efficiency
        float_cols = df.select_dtypes(include=["float64"]).columns
        df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast="float")
        return df

    # Merge datasets for all models
    merged_df = pd.concat(
        (load_and_process(modeltype) for modeltype in MODEL_TYPE),
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

global all_models
all_models = MODEL_TYPE  # df[CATEGORY].unique()


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
        unit (str, optional): The unit to display with the range values. Defaults to an empty string.

    Returns:
        pn.widgets.RangeSlider: A RangeSlider widget initialized with the column's value range.
    """
    slider = pn.widgets.RangeSlider(
        start=table.value[col].min(),
        end=table.value[col].max(),
        format="0.00",
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
    # sliders
    # s_classifier_mean: pn.widgets.RangeSlider,
    columns: list,
    sliders: dict = None,
    categories: list = None,
) -> pn.Column:
    """
    Build an interactive table with ranking and filtering features based on the provided weights and filters.

    Args:
        w_property1 to w_property5: FloatSlider widgets representing weights for each property.
        columns (list): A list of column names to be displayed in the table.
        sliders (dict, optional): A dictionary where keys are column names and values are RangeSlider widgets
            for filtering. Defaults to None.
        categories (list, optional): A list of categories to be displayed. Defaults to None.

    Returns:
        pn.Column: A Panel Column containing the filename input, download button, and the interactive table.
    """
    # Calculate ranking based on weights and normalize
    ranking = (
        w_property1 * min_max_norm(abs(df["Average Band Gap (eV)"] - 1.34))
        + w_property2 * min_max_norm(df["AI-experts confidence (-)"])
        + w_property3 * min_max_norm(df["Formation Energy (eV/atom)"])
        + w_property4 * min_max_norm(df[f"Volume ({ANGSTROM_SYMBOL}{CUBE_SYMBOL})"])
        + w_property5 * min_max_norm(df[f"Density ({ANGSTROM_SYMBOL}{CUBE_SYMBOL}/atom)"])
        + w_property6 * min_max_norm(df["Average Band Gap (deviation) (eV)"])
        + w_property7 * min_max_norm(df["AI-experts confidence (deviation) (-)"])
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
        hidden_models = set(all_models) - set(categories)
        for model in hidden_models:
            table.add_filter(pn.bind(apply_category_filter, category=CATEGORY, item_to_hide=model))

    # Watch sliders and update download button
    def update_download(event):
        # Recreate the download button whenever filters or sliders change
        button = down_load_menu(filename.param.value, table)
        return button

    for slider in sliders.values():
        slider.param.watch(update_download, "value_throttled")

    # Add download section
    filename = pn.widgets.TextInput(name="Enter filename", value="perovskite_candidates.csv")
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
    categories: list = None,
) -> hvplot:
    """
    Builds an interactive scatter plot based on selected filters and model selection.

    Args:
        s_property1-5 (pn.widgets.RangeSlider): Property filters.
        categories (list, optional): List of categories to include in the plot. Default is None.

    Returns:
        hvplot: An interactive scatter plot with applied filters.
    """
    plot_df = df.copy()

    # Apply filters based on sliders
    filters = [
        ("Average Band Gap (eV)", s_property1),
        ("AI-experts confidence (-)", s_property2),
        ("Formation Energy (eV/atom)", s_property3),
        (f"Volume ({ANGSTROM_SYMBOL}{CUBE_SYMBOL})", s_property4),
        (f"Density ({ANGSTROM_SYMBOL}{CUBE_SYMBOL}/atom)", s_property5),
    ]

    for col, slider in filters:
        plot_df = plot_df[(plot_df[col] >= slider[0]) & (plot_df[col] <= slider[1])]

    # Apply filter if provided
    if categories:
        plot_df = plot_df[plot_df[CATEGORY].isin(categories)]

    # Background scatter plot with all data
    back_scatter = df.hvplot.scatter(
        x="AI-experts confidence (-)",
        y="Average Band Gap (eV)",
        s=100,
        alpha=0.25,
        color="#444",
        line_color="white",
        hover_cols="all",
    ).opts(
        tools=[],
        logx=False,
        logy=False,
        xlabel="AI-experts confidence (-)",
        ylabel="Average Band Gap (eV)",
    )

    # Foreground scatter plot with filtered data
    front_scatter = plot_df.hvplot.scatter(
        x="AI-experts confidence (-)",
        y="Average Band Gap (eV)",
        s=100,
        # noqa:E501 hover_cols=['ID', 'Category 1', 'Property 1',
        # 'Property 2', 'Property 3', 'Property 4', 'File'],  hover_cols='all',
        line_color="white",
        c=CATEGORY,
        legend="top",
        hover_cols="all",
    ).opts(
        logx=False,
        logy=False,
        xlabel="AI-experts confidence (-)",
        ylabel="Average Band Gap (eV)",
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
    name="Deviation from ideal Band Gap (eV)",
    start=-10,
    end=10,
    step=0.5,
    value=-1,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property1_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the deviation from ideal Band Gap (eV) in the <b><i>ranking function</i></b>. This deviation is evaluated as the difference between the average predicted Band Gap (eV) and its ideal value (1.34 eV)."
)
weights["Deviation from ideal Band Gap (eV)"] = w_property1
weights_helper["Deviation from ideal Band Gap (eV)"] = w_property1_help
# Property 2
w_property2 = pn.widgets.FloatSlider(
    name="AI-experts confidence (-)",
    start=-10,
    end=10,
    step=0.5,
    value=4,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property2_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'AI-experts confidence (-)' property in the <b><i>ranking function</i></b>."
)
weights["AI-experts confidence (-)"] = w_property2
weights_helper["AI-experts confidence (-)"] = w_property2_help
# Property 3
w_property3 = pn.widgets.FloatSlider(
    name="Formation Energy (eV/atom)",
    start=-10,
    end=10,
    step=0.5,
    value=0,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property3_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'Formation Energy (eV/atom)' property in the <b><i>ranking function</i></b>."
)
weights["Formation Energy (eV/atom)"] = w_property3
weights_helper["Formation Energy (eV/atom)"] = w_property3_help
# Property 4
w_property4 = pn.widgets.FloatSlider(
    name=f"Volume ({ANGSTROM_SYMBOL}{CUBE_SYMBOL})",
    start=-10,
    end=10,
    step=0.5,
    value=0,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property4_help = pn.widgets.TooltipIcon(
    value=f"Adjust the weight of the 'Volume ({ANGSTROM_SYMBOL}{CUBE_SYMBOL})' property in the <b><i>ranking function</i></b>."
)
weights[f"Volume ({ANGSTROM_SYMBOL}{CUBE_SYMBOL})"] = w_property4
weights_helper[f"Volume ({ANGSTROM_SYMBOL}{CUBE_SYMBOL})"] = w_property4_help
# Property 5
w_property5 = pn.widgets.FloatSlider(
    name=f"Density ({ANGSTROM_SYMBOL}{CUBE_SYMBOL}/atom)",
    start=-10,
    end=10,
    step=0.5,
    value=0,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property5_help = pn.widgets.TooltipIcon(
    value=f"Adjust the weight of the 'Density ({ANGSTROM_SYMBOL}{CUBE_SYMBOL}/atom)' property in the <b><i>ranking function</i></b>."
)
weights[f"Density ({ANGSTROM_SYMBOL}{CUBE_SYMBOL}/atom)"] = w_property5
weights_helper[f"Density ({ANGSTROM_SYMBOL}{CUBE_SYMBOL}/atom)"] = w_property5_help
# Property 6
w_property6 = pn.widgets.FloatSlider(
    name="Average Band Gap (deviation) (eV)",
    start=-10,
    end=10,
    step=0.5,
    value=-2,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property6_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'Average Band Gap (deviation) (eV)' property in the <b><i>ranking function</i></b>."
)
weights["Average Band Gap (deviation) (eV)"] = w_property6
weights_helper["Average Band Gap (deviation) (eV)"] = w_property6_help
# Property 7
w_property7 = pn.widgets.FloatSlider(
    name="AI-experts confidence (deviation) (-)",
    start=-10,
    end=10,
    step=0.5,
    value=-2,
    sizing_mode="fixed",
    width=SIDEBAR_WIDGET_W,
)
w_property7_help = pn.widgets.TooltipIcon(
    value="Adjust the weight of the 'AI-experts confidence (deviation) (-)' property in the <b><i>ranking function</i></b>."
)
weights["AI-experts confidence (deviation) (-)"] = w_property7
weights_helper["AI-experts confidence (deviation) (-)"] = w_property7_help


# (2) Widget SIDEBAR : properties range
sliders = {}
sliders_helper = {}

# Property 1
s_property1 = create_range_slider(col="Average Band Gap (eV)", name="Average Band Gap (eV)")
s_property1_help = pn.widgets.TooltipIcon(
    value="<b>Average Band Gap (eV)</b>: Average band gap predicted by the ensemble committee of four E3NN models."
)
sliders["Average Band Gap (eV)"] = s_property1
sliders_helper["Average Band Gap (eV)"] = s_property1_help

# Property 2
s_property2 = create_range_slider(
    col="AI-experts confidence (-)", name="AI-experts confidence (-)"
)
s_property2_help = pn.widgets.TooltipIcon(
    value="<b>AI-experts confidence (-)</b>: Confidence level of the ensemble committee of ten GBDT models in classifying the material as a perovskite."
)
sliders["AI-experts confidence (-)"] = s_property2
sliders_helper["AI-experts confidence (-)"] = s_property2_help

# Property 3
s_property3 = create_range_slider(
    col="Formation Energy (eV/atom)", name="Formation Energy (eV/atom)"
)
s_property3_help = pn.widgets.TooltipIcon(
    value="<b>Formation Energy per Atom (eV/atom)</b>: Measure of the average energy required to form the molecule from its constituent atoms."
)
sliders["Formation Energy (eV/atom)"] = s_property3
sliders_helper["Formation Energy (eV/atom)"] = s_property3_help

# Property 4
s_property4 = create_range_slider(
    col=f"Volume ({ANGSTROM_SYMBOL}{CUBE_SYMBOL})", name=f"Volume ({ANGSTROM_SYMBOL}{CUBE_SYMBOL})"
)
s_property4_help = pn.widgets.TooltipIcon(
    value=f"<b>Volume ({ANGSTROM_SYMBOL}{CUBE_SYMBOL})</b>: Three-dimensional space occupied by the molecule."
)
sliders[f"Volume ({ANGSTROM_SYMBOL}{CUBE_SYMBOL})"] = s_property4
sliders_helper[f"Volume ({ANGSTROM_SYMBOL}{CUBE_SYMBOL})"] = s_property4_help

# Property 5
s_property5 = create_range_slider(
    col=f"Density ({ANGSTROM_SYMBOL}{CUBE_SYMBOL}/atom)",
    name=f"Density ({ANGSTROM_SYMBOL}{CUBE_SYMBOL}/atom)",
)
s_property5_help = pn.widgets.TooltipIcon(
    value=f"<b>Density ({ANGSTROM_SYMBOL}{CUBE_SYMBOL}/atom)</b>: Average space per atom within the molecule."
)
sliders[f"Density ({ANGSTROM_SYMBOL}{CUBE_SYMBOL}/atom)"] = s_property5
sliders_helper[f"Density ({ANGSTROM_SYMBOL}{CUBE_SYMBOL}/atom)"] = s_property5_help


# (3) Widget SIDEBAR: Models selection
select_models = pn.widgets.MultiChoice(
    value=MODEL_ACTIVE,
    options=MODEL_TYPE,
    #  sizing_mode='stretch_width',
    width=SIDEBAR_WIDGET_W,
    sizing_mode="fixed",
    description="Add or remove <i>perovskite</i> candidate datapoints with predictions from a specific <i>model</i>",
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
    # sliders
    # s_classifier_mean=s_classifier_mean,
    columns=select_properties,
    categories=select_models,
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
    categories=select_models,
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

box_select_models = pn.Column(
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
## Regressor models
Easily manage perovskite predictions associated with a specific regressor model."""
        ),
    ),
    select_models,
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
    sidebar=[box_select_models, divider_sb, controls_tabs_intro, controls_tabs],
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
