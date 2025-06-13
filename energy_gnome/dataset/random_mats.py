from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil as sh

from mp_api.client import MPRester
import numpy as np
import pandas as pd
from tqdm import tqdm

from energy_gnome.config import DATA_DIR  # noqa:401
from energy_gnome.dataset.base_dataset import BaseDatabase
from energy_gnome.exception import ImmutableRawDataError
from energy_gnome.utils.logger_config import logger
from energy_gnome.utils.mp_api_utils import (
    convert_my_query_to_dataframe,
    get_mp_api_key,
)

# Fields

MAT_PROPERTIES = {
    "volume": "float64",
    "density": "float64",
    "energy_per_atom": "float64",
    "formation_energy_per_atom": "float64",
    "material_id": "str",
    "formula_pretty": "str",
    "is_stable": "bool",
    "is_magnetic": "bool",
    "band_gap": "float64",
    "is_metal": "bool",
    "last_updated": "str",
    "nsites": "int",
    "elements": "str",
    "nelements": "int",
}

CRITICAL_FIELD = ["band_gap", "is_metal", "material_id", "is_magnetic"]


MP_BATCH_SIZE = 2000


class MPDatabase(BaseDatabase):
    def __init__(self, name: str = "mp", data_dir: Path | str = DATA_DIR):
        """
        Initialize the MPDatabase with a root data directory and processing stage.

        This constructor sets up the directory structure for storing data across different
        processing stages (`raw`, `processed`, `final`). It also initializes placeholders
        for the database paths and data specific to the `MPDatabase` class.

        Args:
            name (str, optional): The name of the database. Defaults to "mp".
            data_dir (Path or str, optional): Root directory path for storing data. Defaults
                to `DATA_DIR` from the configuration.

        Raises:
            NotImplementedError: If the specified processing stage is not supported.
            ImmutableRawDataError: If attempting to set an unsupported processing stage.
        """
        super().__init__(name=name, data_dir=data_dir)

        # Force single directory for raw database of MPDatabase
        self.database_paths["raw"] = self.data_dir / "raw" / "mp" / "database.json"

        self._mp = pd.DataFrame()
        self._is_specialized = False
        self.load_all()

    def retrieve_materials(self, max_framework_size: int = 6, mute_progress_bars: bool = True) -> pd.DataFrame:
        """
        Retrieve and clean materials data from the Materials Project API.

        This method connects to the Materials Project using an API key, iteratively queries
        materials by increasing chemical system complexity (from binary to (`max_framework_size`+1)-element
        systems), and retrieves specified material properties. The results are compiled into a single
        DataFrame, cleaned of missing critical fields, and stored internally.

        Args:
            max_framework_size (int, optional): One less than the maximum number of elements in the chemical
                system to include when querying. For example, a value of 5 will query systems with 2 to 6 elements.
                This defines the upper bound for the number of `*` wildcards used in the chemical system query.
                Defaults to 6.
            mute_progress_bars (bool, optional): If `True`, disables progress bars from the API client
                and other utilities. Useful for cleaner logging in batch scripts or tests. Defaults to True.

        Returns:
            pd.DataFrame: A cleaned and structured DataFrame containing the retrieved materials data.
                Also returns the raw query results (list of material summaries).

        Raises:
            Exception: Propagates any exception raised during API querying (e.g., authentication failure,
                connection issues, or malformed responses).

        Logs:
            - DEBUG: Logs detailed steps of the query and cleaning processes.
            - INFO: Logs progress for each chemical system queried and total results.
            - SUCCESS: Confirms completion of material retrieval.

        Notes:
            - Uses the `get_mp_api_key()` utility to obtain the Materials Project API key.
            - Drops rows missing any values in `CRITICAL_FIELD`, and columns that are entirely NaN.
            - Stores the final DataFrame internally as `self._mp`.

        Example:
            >>> db = SomeCustomDatabase()
            >>> df, raw = db.retrieve_materials(max_framework_size=3)
        """
        mp_api_key = get_mp_api_key()
        logger.debug("MP querying for all materials.")
        query = []

        with MPRester(mp_api_key, mute_progress_bars=mute_progress_bars) as mpr:
            try:
                for n_elm in range(1, max_framework_size + 1):
                    chemsys = "*" + "-*" * n_elm
                    logger.info(f"Retrieving all materials with chemical system = {chemsys} :")
                    query += mpr.materials.summary.search(chemsys=chemsys, fields=MAT_PROPERTIES)
                logger.info(f"MP query successful, {len(query)} materials found.")
            except Exception as e:
                raise e
        logger.debug("Converting MP query results into DataFrame.")
        mp_df = convert_my_query_to_dataframe(query, mute_progress_bars=mute_progress_bars)

        # Fast cleaning
        logger.debug("Removing NaN (rows)")
        logger.debug(f"size DB before = {len(mp_df)}")
        mp_df = mp_df.dropna(axis=0, how="any", subset=CRITICAL_FIELD)
        logger.debug(f"size DB after = {len(mp_df)}")
        logger.debug("Removing NaN (cols)")
        logger.debug(f"size DB before = {len(mp_df)}")
        mp_df = mp_df.dropna(axis=1, how="all")
        logger.debug(f"size DB after = {len(mp_df)}")

        mp_df.reset_index(drop=True, inplace=True)
        self._mp = mp_df.copy()

        logger.success("Materials retrieved successfully.")
        return self._mp, query

    def save_cif_files(
        self,
        stage: str,
        mute_progress_bars: bool = True,
    ) -> None:
        """
        Save CIF files for materials and update the database efficiently.

        This method retrieves crystal structures from the Materials Project API and saves them
        as CIF files in the appropriate directory. It ensures raw data integrity and efficiently
        updates the database with CIF file paths.

        Args:
            stage (str): The processing stage (`raw`, `processed`, `final`).
            mute_progress_bars (bool, optional): If True, disables progress bars. Defaults to True.

        Raises:
            ImmutableRawDataError: If attempting to modify immutable raw data.

        Logs:
            - WARNING: If the CIF directory is being cleaned or if a material ID is missing.
            - ERROR: If an API query fails or CIF file saving encounters an error.
            - INFO: When CIF files are successfully retrieved and saved.
        """

        # Set up directory for saving CIF files
        saving_dir = self.database_directories[stage] / "structures/"
        df = self.get_database(stage)

        # Ensure raw data integrity
        if stage == "raw" and not self._update_raw:
            logger.error("Raw data must be treated as immutable!")
            raise ImmutableRawDataError("Raw data must be treated as immutable!")

        # Clear directory if it exists
        if saving_dir.exists():
            logger.warning(f"Cleaning {saving_dir}")
            sh.rmtree(saving_dir)
        saving_dir.mkdir(parents=True, exist_ok=False)

        # Create a lookup dictionary for material IDs → DataFrame row indices
        material_id_to_index = {mid: idx for idx, mid in enumerate(df["material_id"])}

        # Fetch structures in batches
        ids_list = df["material_id"].tolist()
        n_batch = int(np.ceil(len(ids_list) / MP_BATCH_SIZE))
        mp_api_key = get_mp_api_key()

        logger.debug("MP querying for materials' structures.")
        with MPRester(mp_api_key, mute_progress_bars=mute_progress_bars) as mpr:
            all_cif_paths = {}  # Store updates in a dict to vectorize DataFrame updates later

            for i_batch in tqdm(range(n_batch), desc="Saving materials", disable=mute_progress_bars):
                i_star = i_batch * MP_BATCH_SIZE
                i_end = min((i_batch + 1) * MP_BATCH_SIZE, len(ids_list))

                try:
                    materials_mp_query = mpr.materials.summary.search(
                        material_ids=ids_list[i_star:i_end], fields=["material_id", "structure"]
                    )
                    logger.info(f"MP query successful, {len(materials_mp_query)} structures found.")
                except Exception as e:
                    logger.error(f"Failed MP query: {e}")
                    raise e

                # Define a function to save CIF files in parallel
                def save_cif(material):
                    try:
                        material_id = material.material_id
                        if material_id not in material_id_to_index:
                            logger.warning(f"Material ID {material_id} not found in database.")
                            return None

                        cif_path = saving_dir / f"{material_id}.cif"
                        material.structure.to(filename=str(cif_path))
                        return material_id, str(cif_path)

                    except Exception as e:
                        logger.error(f"Failed to save CIF for {material.material_id}: {e}")
                        return None

                # Parallelize CIF saving (adjust max_workers based on your system)
                with ThreadPoolExecutor(max_workers=None) as executor:
                    results = list(executor.map(save_cif, materials_mp_query))

                # Collect results in dictionary for bulk DataFrame update
                for result in results:
                    if result:
                        material_id, cif_path = result
                        all_cif_paths[material_id] = cif_path

        # Bulk update DataFrame in one step (vectorized)
        df["cif_path"] = df["material_id"].map(all_cif_paths)

        # Save the updated database
        self.save_database(stage)
        logger.info(f"CIF files for stage '{stage}' saved and database updated successfully.")

    def remove_cross_overlap(self, stage: str, df: pd.DataFrame, save_db: bool = False) -> pd.DataFrame:
        """
        Remove entries in the generic database that overlap with a category-specific database.

        This function identifies and removes material entries that exist in both the generic
        and category-specific databases for a given processing stage.

        Args:
            stage (str): Processing stage (`raw`, `processed`, `final`).
            df (pd.DataFrame): The category-specific database to compare with the generic database.

        Returns:
            pd.DataFrame: The filtered generic database with overlapping entries removed.

        Raises:
            ValueError: If an invalid processing stage is provided.

        Logs:
            - ERROR: If an invalid stage is given.
            - INFO: Number of overlapping entries identified and removed.
        """
        if stage not in self.processing_stages:
            logger.error(f"Invalid stage: {stage}. Must be one of {self.processing_stages}.")
            raise ValueError(f"stage must be one of {self.processing_stages}.")

        self.load_database(stage)
        mp_df = self.get_database(stage)
        if "material_id" in df.columns:
            id_overlap = df["material_id"].tolist()
            mp_df["to_drop"] = mp_df.apply(lambda x: x["material_id"] in id_overlap, axis=1)
        else:
            formula_overlap = df["formula_pretty"].tolist()
            mp_df["to_drop"] = mp_df.apply(lambda x: x["formula_pretty"] in formula_overlap, axis=1)

        to_drop = mp_df["to_drop"].value_counts().get(True, 0)
        logger.info(f"{to_drop} overlapping items to drop.")
        mp_df_no_overlap = mp_df.iloc[np.where(mp_df["to_drop"] == 0)[0], :].reset_index(drop=True)
        mp_df_no_overlap = mp_df_no_overlap.drop(columns=["to_drop"])

        if save_db:
            self.databases[stage] = mp_df_no_overlap.copy()

        return mp_df_no_overlap

    def __repr__(self) -> str:
        """
        Text representation of the MPDatabase instance.
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
        HTML representation of the MPDatabase instance.
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
