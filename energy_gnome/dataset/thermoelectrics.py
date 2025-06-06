from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from energy_gnome.config import DATA_DIR, EXTERNAL_DATA_DIR  # noqa:401
from energy_gnome.dataset.base_dataset import BaseDatabase
from energy_gnome.dataset.utils import normalized_formula, recompose_df
from energy_gnome.exception import ImmutableRawDataError
from energy_gnome.utils.logger_config import logger

# Paths
ELEMENTS = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Uut",
    "Fl",
    "Uup",
    "Lv",
    "Uus",
    "Uuo",
]


class ThermoelectricDatabase(BaseDatabase):
    def __init__(
        self,
        name: str = "thermoeletrics",
        data_dir: Path | str = DATA_DIR,
    ):
        """
        Initialize the ThermoelectricDatabase with a root data directory and processing stage.

        This constructor sets up the directory structure for storing data across different
        processing stages (`raw`, `processed`, `final`). It also initializes placeholders
        for the database paths and data specific to the `ThermoelectricDatabase` class, including
        a path for the external Thermoelectric project data.

        Args:
            name (str, optional): The name of the database. Defaults to "thermoelectrics".
            data_dir (Path or str, optional): Root directory path for storing data. Defaults
                to `DATA_DIR` from the configuration.

        Raises:
            NotImplementedError: If the specified processing stage is not supported.
            ImmutableRawDataError: If attempting to set an unsupported processing stage.
        """
        super().__init__(name=name, data_dir=data_dir)
        self._thermoelectrics = pd.DataFrame()
        self.external_estm_path: Path | str = (
            self.data_dir / Path(EXTERNAL_DATA_DIR) / Path("thermoelectrics") / Path("estm.xlsx")
        )
        self._is_specialized = True
        self.load_all()

    def retrieve_materials(self) -> pd.DataFrame:
        """
        Retrieve thermoelectric material data from the external ESTM Excel file.

        This method reads an external Excel spreadsheet containing experimentally synthesized
        thermoelectric materials (ESTM) data. It extracts only the relevant columns:
        chemical formula (`Formula`), measurement temperature in Kelvin (`temperature(K)`),
        and thermoelectric figure of merit (`ZT`).

        Returns:
            pd.DataFrame: A DataFrame with columns ['Formula', 'temperature(K)', 'ZT'] containing
                the ESTM data.

        Notes:
            - The external ESTM file path is defined during initialization of the `ThermoelectricDatabase`.
            - This method assumes the Excel file exists and contains the specified columns.
        """
        logger.debug("Reading ESTM database file.")
        data = pd.read_excel(self.external_estm_path)
        data = data[["Formula", "temperature(K)", "ZT"]]

        return data

    def compare_and_update(self, new_db: pd.DataFrame, stage: str) -> pd.DataFrame:
        """
        Compare and update the database with new entries.

        This method checks for new entries in the provided database and updates the stored
        database accordingly. It ensures that raw data remains immutable unless explicitly
        allowed. If new entries are found, the old database is backed up, and a changelog
        is created.

        Args:
            new_db (pd.DataFrame): The new database to compare against the existing one.
            stage (str): The processing stage ("raw", "processed", "final").

        Returns:
            pd.DataFrame: The updated database containing new entries.

        Raises:
            ImmutableRawDataError: If attempting to modify raw data without explicit permission.

        Logs:
            - WARNING: If new items are detected in the database.
            - ERROR: If an attempt is made to modify immutable raw data.
            - INFO: When saving or updating the database.
            - INFO: If no new items are found and no update is required.
        """
        old_db = self.get_database(stage=stage)
        old_db.rename(columns={"Formula": "material_id"}, inplace=True)
        new_db.rename(columns={"Formula": "material_id"}, inplace=True)

        db_diff = self.compare_databases(new_db, stage)

        if not db_diff.empty:
            logger.warning(f"The new database contains {len(db_diff)} new items.")

            if stage == "raw" and not self._update_raw:
                logger.error("Raw data must be treated as immutable!")
                logger.error(
                    "It's okay to read and copy raw data to manipulate it into new outputs, but never okay to change it in place."
                )
                raise ImmutableRawDataError(
                    "Raw data must be treated as immutable!\n"
                    "It's okay to read and copy raw data to manipulate it into new outputs, but never okay to change it in place."
                )
            else:
                if stage == "raw":
                    logger.info("Be careful you are changing the raw data which must be treated as immutable!")
                if old_db.empty:
                    logger.info(f"Saving new {stage} data in {self.database_paths[stage]}.")
                else:
                    logger.info(f"Updating the {stage} data and saving it in {self.database_paths[stage]}.")
                    self.backup_and_changelog(
                        old_db,
                        new_db,
                        db_diff,
                        stage,
                    )

                new_db.rename(columns={"material_id": "Formula"}, inplace=True)

                self.databases[stage] = new_db.copy()
                self.save_database(stage)
        else:
            new_db.rename(columns={"material_id": "Formula"}, inplace=True)
            logger.info("No new items found. No update required.")

    def process_database(
        self,
        inplace: bool = True,
        df: pd.DataFrame = None,
        temp_list: list[float] = list(np.arange(300, 1000, 130, float)),
    ) -> pd.DataFrame:
        """
        Process and structure thermoelectric materials data for machine learning or analysis.

        This method supports two processing modes:
        - If `inplace=True`, it loads and processes the internal "raw" database by normalizing chemical formulas,
        grouping rows with identical formulas, and aggregating associated properties (`ZT`, `temperature(K)`,
        and `is_specialized`) into lists. The recomposed data is saved to the "processed" stage.
        - If `inplace=False`, the method takes an external DataFrame, ensures a `temperature(K)` column exists
        using a provided temperature list, and explodes this column to generate a row per temperature point.

        Args:
            inplace (bool, optional): Whether to operate on the internal database (True) or return a processed
                version of the input DataFrame (False). Defaults to True.
            df (pd.DataFrame, optional): Raw data to be processed when `inplace=False`. Must contain at least a
                'Formula' column. Defaults to None.
            temp_list (list[float], optional): List of temperatures (in Kelvin) to insert if the input DataFrame
                lacks a 'temperature(K)' column. Used only when `inplace=False`. Defaults to range(300, 1000, 130).

        Returns:
            pd.DataFrame: If `inplace=False`, returns the processed and structured DataFrame. If `inplace=True`,
                stores the result internally and returns None.

        Raises:
            ValueError: If `inplace` is False and no input DataFrame (`df`) is provided.

        Notes:
            - When `inplace=True`, the method assumes the raw database contains columns:
            ['Formula', 'ZT', 'temperature(K)', 'is_specialized'].
            - Chemical formulas are normalized using the `normalized_formula` utility.
            - In `inplace=False` mode, the method supports augmentation by adding and exploding temperature values.
        """
        if not inplace and df is None:
            logger.error("Invalid input: You must input a pd.DataFrame if 'inplace' is set to False.")
            raise ValueError("You must input a pd.DataFrame if 'inplace' is set to False.")

        if inplace:
            self.load_database(stage="raw")
            raw_df = self.get_database(stage="raw")
        else:
            raw_df = df

        data = raw_df.copy()
        if inplace:
            data["formula_pretty"] = ""

            logger.debug("Normalizing formulae.")
            for i in tqdm(range(len(data))):
                data.loc[i, "formula_pretty"] = normalized_formula(data["Formula"].iloc[i])

            grouped_df = data.groupby("formula_pretty")
            grouped_ZT = grouped_df["ZT"].apply(list).reset_index()
            grouped_temperature = grouped_df["temperature(K)"].apply(list).reset_index()
            grouped_specialized = grouped_df["is_specialized"].apply(list).reset_index()
            grouped_temperature["ZT"] = grouped_ZT["ZT"]
            grouped_temperature["is_specialized"] = grouped_specialized["is_specialized"]

            logger.debug("Recomposing dataframe.")
            df_recomposed = recompose_df(grouped_temperature)
            df_recomposed.reset_index(inplace=True, drop=True)
        else:
            logger.debug("Inserting temperature values.")
            if "temperature(K)" not in raw_df.columns:
                raw_df.insert(len(raw_df.columns) - 1, "temperature(K)", [temp_list] * len(raw_df))
            df_recomposed = raw_df.explode("temperature(K)").reset_index(drop=True)

        if inplace:
            self.databases["processed"] = df_recomposed.copy()
            self.save_database("processed")
        else:
            return df_recomposed

    def __repr__(self) -> str:
        """
        Text representation of the ThermoelectricDatabase instance.
        Used for `print()` and `str()` calls.

        Returns:
            str: ASCII table representation of the database
        """
        # Gather information about each stage
        data = {
            "Stage": [],
            "Entries": [],
            "Last Modified": [],
            "Size": [],
            "Storage Path": [],
        }

        # Calculate column widths
        widths = [10, 8, 17, 10, 55]

        for stage in self.processing_stages:
            # Get database info
            db = self.databases[stage]
            path = self.database_paths[stage]

            # Get file modification time and size if file exists
            if path.exists():
                modified = path.stat().st_mtime
                modified_time = pd.Timestamp.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M")
                size = path.stat().st_size / 1024  # Convert to KB
                size_str = f"{size:.1f} KB" if size < 1024 else f"{size / 1024:.1f} MB"
            else:
                modified_time = "Not created"
                size_str = "0 KB"

            path_str = str(path.resolve())
            if len(path_str) > widths[4]:
                path_str = ".." + path_str[len(path_str) - widths[4] + 3 :]

            # Append data
            data["Stage"].append(stage.capitalize())
            data["Entries"].append(len(db))
            data["Last Modified"].append(modified_time)
            data["Size"].append(size_str)
            data["Storage Path"].append(path_str)

        # Create DataFrame
        info_df = pd.DataFrame(data)

        # Text representation for terminal/print
        def create_separator(widths):
            return "+" + "+".join("-" * (w + 1) for w in widths) + "+"

        # Create the text representation
        lines = []

        # Add title
        title = f" {self.__class__.__name__} Summary "
        lines.append(f"\n{title:=^{sum(widths) + len(widths) * 2 + 1}}")

        # Add header
        separator = create_separator(widths)
        lines.append(separator)

        header = "|" + "|".join(f" {col:<{widths[i]}}" for i, col in enumerate(info_df.columns)) + "|"
        lines.append(header)
        lines.append(separator)

        # Add data rows
        for _, row in info_df.iterrows():
            line = "|" + "|".join(f" {str(val):<{widths[i]}}" for i, val in enumerate(row)) + "|"
            lines.append(line)

        # Add bottom separator
        lines.append(separator)

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """
        HTML representation of the ThermoelectricDatabase instance.
        Used for Jupyter notebook display.

        Returns:
            str: HTML representation of the database
        """
        # Gather information about each stage
        data = {
            "Stage": [],
            "Entries": [],
            "Last Modified": [],
            "Size": [],
            "Storage Path": [],
        }

        for stage in self.processing_stages:
            # Get database info
            db = self.databases[stage]
            path = self.database_paths[stage]

            # Get file modification time and size if file exists
            if path.exists():
                modified = path.stat().st_mtime
                modified_time = pd.Timestamp.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M")
                size = path.stat().st_size / 1024  # Convert to KB
                size_str = f"{size:.1f} KB" if size < 1024 else f"{size / 1024:.1f} MB"
            else:
                modified_time = "Not created"
                size_str = "0 KB"

            # Append data
            data["Stage"].append(stage.capitalize())
            data["Entries"].append(len(db))
            data["Last Modified"].append(modified_time)
            data["Size"].append(size_str)
            data["Storage Path"].append(str(path.resolve()))

        # Create DataFrame
        info_df = pd.DataFrame(data)

        # Generate header row
        header_cells = " ".join(
            f'<th style="padding: 12px 15px; text-align: left;">{col}</th>' for col in info_df.columns
        )

        # Generate table rows
        table_rows = ""
        for _, row in info_df.iterrows():
            cells = "".join(f'<td style="padding: 12px 15px;">{val}</td>' for val in row)
            table_rows += f"<tr style='border-bottom: 1px solid #e9ecef;'>{cells}</tr>"

        # Create the complete HTML
        html = (
            """<style>
                @media (prefers-color-scheme: dark) {
                    .database-container { background-color: #1e1e1e !important; }
                    .database-title { color: #e0e0e0 !important; }
                    .database-table { background-color: #2d2d2d !important; }
                    .database-header { background-color: #4a4a4a !important; }
                    .database-cell { border-color: #404040 !important; }
                    .database-info { color: #b0b0b0 !important; }
                }
            </style>"""
            '<div style="font-family: Arial, sans-serif; padding: 20px; background:transparent; '
            'border-radius: 8px;">'
            f'<h3 style="color: #58bac7; margin-bottom: 15px;">{self.__class__.__name__}</h3>'
            '<div style="overflow-x: auto;">'
            '<table class="database-table" style="border-collapse: collapse; width: 100%;'
            ' box-shadow: 0 1px 3px rgba(0,0,0,0.1); background:transparent;">'
            # '<table style="border-collapse: collapse; width: 100%; background-color: white; '
            # 'box-shadow: 0 1px 3px rgba(0,0,0,0.1);">'
            "<thead>"
            f'<tr style="background-color: #58bac7; color: white;">{header_cells}</tr>'
            "</thead>"
            f"<tbody>{table_rows}</tbody>"
            "</table>"
            "</div>"
            '<div style="margin-top: 10px; color: #666; font-size: 1.1em;">'
            "</div>"
            "</div>"
        )
        return html
