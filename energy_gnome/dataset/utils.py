import os
from pathlib import Path

from ase.io import read
from loguru import logger
import matplotlib.pyplot as plt
from monty.fractions import gcd_float
import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from energy_gnome.utils.readers import to_unix

tqdm.pandas()


def make_link(source: Path, target: Path) -> None:
    """
    Creates a symbolic link from the source to the target.

    This function creates a symbolic link from the `source` path to the `target` path.
    If the target path already exists, it is overwritten by the new symbolic link.
    Symbolic links are useful for creating references to files and directories
    without duplicating data.

    Args:
        source (Path): The path to the source file or directory.
        target (Path): The path where the symbolic link will be created.

    Returns:
        None

    Raises:
        OSError: If the symbolic link cannot be created due to permission issues
            or invalid paths.

    Logs:
        INFO: When the symbolic link is successfully created.
        ERROR: If the symbolic link creation fails.
    """
    if source.exists():
        logger.warning(f"File {source} already exist")
    else:
        os.symlink(source, target)
        logger.info(f"Made link {source} -> {target}")


def detect_outliers_iqr(df: pd.DataFrame, iqr_w: float = 1.5):
    """Detects outliers in a Pandas DataFrame using the IQR method.

    Args:
        df (DataFrame): Pandas DataFrame with numerical data.
        iqr_w (float): IQR width, default value 1.4

    Returns:
        _type_: A Pandas DataFrame of the same shape as `df` with True for outliers and False otherwise.
    """

    outlier_mask = pd.DataFrame(data=False, index=df.index, columns=df.columns)

    for col in df.columns:
        if df[col].dtype.kind in "ifc":  # Check if the column type is numeric (boolean, integer, float, complex)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask[col] = (df[col] < (Q1 - iqr_w * IQR)) | (df[col] > (Q3 + iqr_w * IQR))

    return outlier_mask


def get_ordered_integer_formula(el_amt, max_denominator=1000):
    """Converts a mapping of {element: stoichiometric value} to a alphabetically ordered string.

    Given a dictionary of {element : stoichiometric value, ..}, returns a string with
    elements ordered alphabetically and stoichiometric values normalized to smallest common
    integer denominator.

    Args:
        el_amt: {element: stoichiometric value} mapping.
        max_denominator: The maximum common denominator of stoichiometric values to use for
            normalization. Smaller stoichiometric fractions will be converted to the same
            integer stoichiometry.

    Returns:
        A material formula string with elements ordered alphabetically and the stoichiometry
        normalized to the smallest integer fractions.
    """
    g = gcd_float(list(el_amt.values()), 1 / max_denominator)
    d = {k: round(v / g) for k, v in el_amt.items()}
    formula = ""
    for k in sorted(d):
        if d[k] > 1:
            formula += k + str(d[k])
        elif d[k] != 0:
            formula += k
    return formula


def normalized_formula(formula, max_denominator=1000):
    """Normalizes chemical formula to smallest common integer denominator, and orders elements alphabetically.

    Args:
        formula: the string formula.
        max_denominator: highest precision for the denominator (1000 by default).

    Returns:
        A normalized formula string, e.g. Ni0.5Fe0.5 -> FeNi.
    """
    formula_dict = Composition(formula).get_el_amt_dict()
    return get_ordered_integer_formula(formula_dict, max_denominator)


def recompose_df(df):

    new_data = {"formula_pretty": [], "temperature(K)": [], "ZT": [], "is_specialized": []}
    for index, row in df.iterrows():
        formula = row["formula_pretty"]
        temperatures = row["temperature(K)"]
        zt_values = row["ZT"]
        special = row["is_specialized"]

        for temperature, zt_value, special in zip(temperatures, zt_values, special):
            new_data["formula_pretty"].append(formula)
            new_data["temperature(K)"].append(temperature)
            new_data["ZT"].append(zt_value)
            new_data["is_specialized"].append(special)

    new_df = pd.DataFrame(new_data)
    new_df = new_df.groupby(["formula_pretty", "temperature(K)", "is_specialized"])["ZT"].apply(list).reset_index()

    new_df["rsd"] = 0.0
    for i in range(len(new_df)):
        new_df.loc[i, "rsd"] = 100 * np.std(new_df["ZT"].iloc[i]) / np.mean(new_df["ZT"].iloc[i])
        new_df.loc[i, "ZT"] = np.mean(new_df["ZT"].iloc[i])

    new_df = new_df[new_df["rsd"] < 20]
    new_df = new_df.iloc[:, :-1]

    return new_df


def get_element_statistics(df: pd.DataFrame, species: list[str]) -> pd.DataFrame:
    """
    Generates a DataFrame containing statistics of elements present in a given DataFrame.

    This function processes a DataFrame that contains information about different samples,
    specifically their species (elemental composition). It returns a DataFrame that summarizes
    how many samples contain each element and the indices of those samples.

    Args:
        df (pd.DataFrame): A DataFrame containing information about various samples. It is expected
                           to have a column 'species' which lists elements present in each sample.
        species (list[str]): A list of unique element symbols (species) to be analyzed.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to an element. It includes columns for
                      the element symbol ('symbol'), the indices of samples containing the element ('data'),
                      and the count of such samples ('count').
    """
    # Create dictionary indexed by element names storing index of samples containing given element
    species_dict = {k: [] for k in species}
    for entry in df.itertuples():
        for specie in entry.species:
            species_dict[specie].append(entry.Index)

    # Create DataFrame of element statistics
    stats = pd.DataFrame({"symbol": species})
    stats["data"] = stats["symbol"].astype("object")
    for specie in species:
        index_value = stats.index[stats["symbol"] == specie].values[0]
        stats.at[index_value, "data"] = species_dict[specie]
    stats["count"] = stats["data"].apply(len)

    return stats


def split_data(df: pd.DataFrame, test_size: float, seed: int) -> tuple[list, list]:
    """
    Splits a DataFrame into training and testing indices based on a specified test size and random seed.

    This function processes a DataFrame where each row corresponds to a different element or species,
    with a column 'data' containing the indices of samples containing that element. It splits these indices
    into training and testing sets while ensuring that samples are distributed across different elements.

    Args:
        df (pd.DataFrame): The DataFrame to be split. Expected to contain a 'data' column with lists of indices.
        test_size (float): The proportion of the dataset to include in the test split.
        seed (int): The random seed used for creating reproducible splits.

    Returns:
        tuple[list, list]: Two lists containing the indices for the training and testing sets, respectively.
    """
    # Initialize output arrays
    idx_train, idx_test = [], []

    # Remove empty examples and sort df in order of fewest to most examples
    df = df[df["data"].str.len() > 0].sort_values("count")

    for _, entry in tqdm(df.iterrows(), total=len(df), bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"):
        df_specie = entry.to_frame().T.explode("data")

        try:
            # Attempt to split data for the current species
            idx_train_s, idx_test_s = train_test_split(df_specie["data"].values, test_size=test_size, random_state=seed)
        except:  # noqa: E722
            # If split fails (e.g., too few examples), skip this iteration
            pass
        else:
            # Add new examples that do not exist in previous lists
            idx_train += [k for k in idx_train_s if k not in idx_train + idx_test]
            idx_test += [k for k in idx_test_s if k not in idx_train + idx_test]

    return idx_train, idx_test


def element_representation(x: list, idx: list) -> float:
    """
    Calculates the fraction of samples in a dataset that contain a given element.

    This function computes the proportion of samples in the list `x` that contain an element
    specified by the indices in `idx`.

    Args:
        x (list): A list of indices or elements representing the dataset.
        idx (list): A list of indices or elements representing a specific element.

    Returns:
        float: The fraction of samples in `x` that contain the element specified by `idx`.
    """
    # Calculate the fraction of samples containing the given element
    return len([k for k in x if k in idx]) / len(x) if x else 0


def extract_species_from_formula(formula: str) -> list[str]:
    try:
        comp = Composition(formula)
        return list(comp.elements[i].symbol for i in range(len(comp.elements)))
    except Exception as e:
        logger.warning(f"Failed to parse formula '{formula}': {e}")
        return []


def split_subplot(
    ax: plt.Axes,
    df: pd.DataFrame,
    species: list[str],
    dataset: str,
    bottom: float = 0.0,
    legend: bool = False,
) -> None:
    """
    Plots element representation bars on a given Axes object for a specific dataset.

    This function takes an Axes object and plots a bar graph on it representing the
    fraction of samples containing each element from a list of species. The data for
    the bar graph is taken from a specified column in a DataFrame.

    Args:
        ax (plt.Axes): A Matplotlib Axes object where the bar graph will be plotted.
        df (pd.DataFrame): A DataFrame containing the data to be plotted.
        species (list[str]): A list of element symbols (species) to be included in the plot.
        dataset (str): The column name in df which contains the data to be plotted.
        bottom (float, optional): The starting value for the bottom of the bars (used for stacked bars). Default is 0.
        legend (bool, optional): If True, display a legend on the plot. Default is False.

    Returns:
        None: The function modifies the given Axes object but does not return anything.
    """
    COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # Plot element representation
    width = 0.4
    datasets = ["train", "valid", "test"]
    colors = dict(zip(datasets, COLORS))
    color = [int(colors[dataset].lstrip("#")[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
    bx = np.arange(len(species))

    ax.bar(bx, df[dataset], width, fc=color + [0.7], ec=color, lw=1.5, bottom=bottom, label=dataset)

    ax.set_xticks(bx)
    ax.set_xticklabels(species)
    ax.tick_params(direction="in", length=0, width=1)
    ax.set_ylim(top=1.18)
    if legend:
        ax.legend(frameon=False, ncol=3, loc="upper left")


def train_valid_test_split(
    df: pd.DataFrame,
    species: list[str],
    valid_size: float,
    test_size: float,
    datasets: list[str] = ["train", "valid", "test"],
    seed: int = 12,
    plot: bool = False,
) -> tuple[list, list, list]:
    """
    Performs an element-balanced train/validation/test split on a DataFrame and optionally plots the element
    representation.

    This function splits a DataFrame into training, validation, and test sets in a way that balances the representation
    of different elements (species) across these sets. It can also plot the distribution of elements in each set.

    Args:
        df (pd.DataFrame): The DataFrame to be split.
        species (list[str]): A list of unique element symbols (species) to be analyzed.
        valid_size (float): The proportion of the dataset to include in the validation split.
        test_size (float): The proportion of the dataset to include in the test split.
        datasets (list[str], optional): Names of the datasets to be used in plotting.
            Default is ['train', 'valid', 'test'].
        seed (int, optional): The random seed used for creating reproducible splits. Default is 12.
        plot (bool, optional): If True, plots the element representation in each dataset. Default is False.

    Returns:
        tuple[list, list, list]: Three lists containing the indices for the training, validation, and
                                 testing sets, respectively.
    """
    # Perform an element-balanced train/validation/test split
    dev_size = valid_size + test_size
    if abs(dev_size - test_size) < 1e-8:
        logger.debug("split train/test ...")
        stats = get_element_statistics(df, species)
        idx_train, idx_test = split_data(stats, test_size, seed)
        idx_valid = []
        # idx_train += df[~df.index.isin(idx_train + idx_valid + idx_test)].index.tolist()
    elif abs(dev_size - valid_size) < 1e-8:
        logger.debug("split train/valid ...")
        stats = get_element_statistics(df, species)
        idx_train, idx_valid = split_data(stats, valid_size, seed)
        idx_test = []
    else:
        logger.debug("split train/dev ...")
        dev_size = valid_size + test_size
        stats = get_element_statistics(df, species)
        idx_train, idx_dev = split_data(stats, dev_size, seed)

        logger.debug("split valid/test ...")
        stats_dev = get_element_statistics(df.iloc[idx_dev], species)
        idx_valid, idx_test = split_data(stats_dev, test_size / dev_size, seed)
        idx_train += df[~df.index.isin(idx_train + idx_valid + idx_test)].index.tolist()

    # Print dataset sizes and assert no overlap
    logger.debug(f"number of training examples: {len(idx_train)}")
    logger.debug(f"number of validation examples: {len(idx_valid)}")
    logger.debug(f"number of testing examples: {len(idx_test)}")
    logger.debug(f"total number of examples: {len(idx_train + idx_valid + idx_test)}")
    assert len(set.intersection(*map(set, [idx_train, idx_valid, idx_test]))) == 0

    # Optionally plot element representation in each dataset
    if plot:
        stats["train"] = stats["data"].map(lambda x: element_representation(x, np.sort(idx_train)))
        stats["valid"] = stats["data"].map(lambda x: element_representation(x, np.sort(idx_valid)))
        stats["test"] = stats["data"].map(lambda x: element_representation(x, np.sort(idx_test)))
        stats = stats.sort_values("symbol")

        fig, ax = plt.subplots(2, 1, figsize=(14, 7))
        b0, b1 = 0.0, 0.0
        for i, dataset in enumerate(datasets):
            split_subplot(
                ax[0],
                stats[: len(stats) // 2],
                species[: len(stats) // 2],
                dataset,
                bottom=b0,
                legend=True,
            )
            split_subplot(ax[1], stats[len(stats) // 2 :], species[len(stats) // 2 :], dataset, bottom=b1)

            b0 += stats.iloc[: len(stats) // 2][dataset].values
            b1 += stats.iloc[len(stats) // 2 :][dataset].values

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.1)

    return idx_train, idx_valid, idx_test


def random_split(dataset: pd.DataFrame, target_property: str, valid_size=0.2, test_size=0.05, seed=42):
    """
    Perform a random train-validation-test split with specified sizes.
    If structure data is not available, fall back to using the 'formula_pretty' column.

    Args:
        dataset (pd.DataFrame): Input dataset.
        target_property (str): Target property column name.
        valid_size (float, optional): Validation set fraction. Defaults to 0.2.
        test_size (float, optional): Test set fraction. Defaults to 0.05.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        dict: Dictionary with 'train', 'valid', and 'test' DataFrames.
    """

    if "cif_path" in dataset.columns:
        database_split = pd.DataFrame(
            {
                "structure": pd.Series(dtype="object"),
                "species": pd.Series(dtype="object"),
            }
        )

        # Read all structures in parallel using `apply`
        database_split["structure"] = dataset["cif_path"]
        database_split["structure"] = database_split["structure"].progress_apply(lambda path: read(Path(to_unix(path))))

        # Extract properties in bulk
        database_split["formula"] = database_split["structure"].progress_apply(lambda s: s.get_chemical_formula())
        database_split["species"] = database_split["structure"].progress_apply(
            lambda s: list(set(s.get_chemical_symbols()))
        )
    elif "formula_pretty" in dataset.columns:
        database_split = pd.DataFrame()
        database_split["formula"] = dataset["formula_pretty"]
        database_split["species"] = database_split["formula"].progress_apply(extract_species_from_formula)
    else:
        raise ValueError("Dataset must contain either 'cif_path' or 'Formula' column.")

    # Ensure target_property is preserved
    database_split[target_property] = dataset[target_property]

    # Determine unique species across the dataset
    unique_species = sorted(set(sum(database_split["species"].tolist(), [])))

    # Perform train/valid/test split
    idx_train, idx_valid, idx_test = train_valid_test_split(
        database_split,
        unique_species,
        valid_size=valid_size,
        test_size=test_size,
        seed=seed,
        plot=False,
    )

    # Create the final dataset dictionary
    database_dict = {
        "train": dataset.iloc[idx_train],
        "valid": dataset.iloc[idx_valid],
        "test": dataset.iloc[idx_test],
    }

    return database_dict
