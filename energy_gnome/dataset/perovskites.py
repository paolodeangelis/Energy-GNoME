from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import shutil as sh

from mp_api.client import MPRester
import pandas as pd
from tqdm import tqdm

from energy_gnome.config import DATA_DIR, EXTERNAL_DATA_DIR, RAW_DATA_DIR  # noqa:401
from energy_gnome.dataset.base_dataset import BaseDatabase
from energy_gnome.exception import ImmutableRawDataError, MissingData
from energy_gnome.utils.logger_config import logger
from energy_gnome.utils.mp_api_utils import (
    convert_my_query_to_dataframe,
    get_mp_api_key,
)

# Paths
PEROVSKITES_RAW_DATA_DIR = RAW_DATA_DIR / "perovskites"

# Fields

BAND_FIELDS = [
    "band_gap",
    "cbm",
    "vbm",
    "efermi",
    "is_gap_direct",
    "is_metal",
    "magnetic_ordering",
    "nsites",
    "elements",
    "nelements",
    "volume",
    "density",
    "density_atomic",
    "symmetry",
    "material_id",
    "dos",
]

BAND_CRITICAL_FIELD = ["band_gap", "is_metal", "material_id", "is_magnetic"]

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


class PerovskiteDatabase(BaseDatabase):
    def __init__(
        self,
        name: str = "perovskites",
        data_dir: Path | str = DATA_DIR,
        external_perovproj_path: Path | str = EXTERNAL_DATA_DIR / Path("perovskites") / Path("perovproject_db.json"),
    ):
        """
        Initialize the PerovskiteDatabase with a root data directory and processing stage.

        This constructor sets up the directory structure for storing data across different
        processing stages (`raw`, `processed`, `final`). It also initializes placeholders
        for the database paths and data specific to the `PerovskiteDatabase` class, including
        a path for the external Perovskite project data.

        Args:
            name (str, optional): The name of the database. Defaults to "perovskites".
            data_dir (Path or str, optional): Root directory path for storing data. Defaults
                to `DATA_DIR` from the configuration.
            external_perovproj_path (Path or str, optional): Path to the external Perovskite
                Project database file. Defaults to `EXTERNAL_DATA_DIR / "perovskites" /
                "perovproject_db.json"`.

        Raises:
            NotImplementedError: If the specified processing stage is not supported.
            ImmutableRawDataError: If attempting to set an unsupported processing stage.
        """
        super().__init__(name=name, data_dir=data_dir)
        self._perovskites = pd.DataFrame()
        self.external_perovproj_path: Path | str = external_perovproj_path
        self._is_specialized = True

    def _pre_retrieve_robo(self, mute_progress_bars: bool = True) -> list[str]:
        """
        Retrieve Perovskite material IDs from the Robocrystallographer API.

        This method queries the Robocrystallographer tool through the Materials Project
        API to search for materials related to "Perovskite". It returns a list of material
        IDs corresponding to the Perovskite materials identified in the search results.

        Args:
            mute_progress_bars (bool, optional): Whether to mute progress bars for the
                API request. Defaults to `True`.

        Returns:
            (list[str]): A list of material IDs for Perovskite materials retrieved from
            Robocrystallographer.

        Raises:
            Exception: If there is an issue with the query to the Materials Project API.

        Logs:
            - INFO: If the query is successful, logging the number of Perovskite IDs found.
        """
        mp_api_key = get_mp_api_key()
        with MPRester(mp_api_key, mute_progress_bars=mute_progress_bars) as mpr:
            try:
                query = mpr.materials.robocrys.search(keywords=["Perovskite", "perovskite"])
                logger.info(f"MP query successful, {len(query)} perovskite IDs found through Robocrystallographer.")
            except Exception as e:
                raise e
        ids_list_robo = [q.material_id for q in query]
        return ids_list_robo

    def _pre_retrieve_perovproj(self, mute_progress_bars: bool = True) -> list[str]:
        """
        Retrieve Perovskite material IDs from the Perovskite Project.

        This method queries the Materials Project API using the formulae from an external
        Perovskite Project file to search for matching material IDs. It returns a list of
        material IDs for Perovskite materials found through the Perovskite Project.

        Args:
            mute_progress_bars (bool, optional): Whether to mute progress bars for the
                API request. Defaults to `True`.

        Returns:
            (list[str]): A list of material IDs for Perovskite materials retrieved from
            the Perovskite Project.

        Raises:
            Exception: If there is an issue with the query to the Materials Project API.

        Logs:
            - INFO: If the query is successful, logging the number of Perovskite IDs found.
        """
        mp_api_key = get_mp_api_key()
        with open(self.external_perovproj_path) as f:
            dict_ = json.load(f)
        with MPRester(mp_api_key, mute_progress_bars=mute_progress_bars) as mpr:
            try:
                query = mpr.materials.summary.search(formula=dict_, fields="material_id")
                logger.info(
                    f"MP query successful, {len(query)} perovskite IDs found through Perovskite Project formulae."
                )
            except Exception as e:
                raise e
        ids_list_perovproj = [q.material_id for q in query]
        return ids_list_perovproj

    def retrieve_materials(self, mute_progress_bars: bool = True) -> pd.DataFrame:
        """
        Retrieve Perovskite materials from the Materials Project API.

        This method connects to the Materials Project API using the `MPRester`, queries
        for materials related to Perovskites, and retrieves specified properties. The data
        is then cleaned by removing rows with missing critical fields and filtering out
        metallic perovskites. The method returns a cleaned DataFrame of Perovskites.

        Args:
            mute_progress_bars (bool, optional):
                If `True`, mutes the Material Project API progress bars during the request.
                Defaults to `True`.

        Returns:
            (pd.DataFrame): A DataFrame containing the retrieved and cleaned Perovskite materials.

        Raises:
            Exception: If the API query fails or any issue occurs during data retrieval.

        Logs:
            - DEBUG: Logs the process of querying and cleaning data.
            - INFO: Logs successful query results and how many Perovskites were retrieved.
            - SUCCESS: Logs the successful retrieval of Perovskite materials.
        """
        mp_api_key = get_mp_api_key()
        ids_list_robo = self._pre_retrieve_robo(mute_progress_bars=mute_progress_bars)
        ids_list_perovproj = self._pre_retrieve_perovproj(mute_progress_bars=mute_progress_bars)
        logger.debug("MP querying for perovskites.")

        ids_list = ids_list_robo + ids_list_perovproj
        unique_ids = list()
        for x in ids_list:
            if x not in unique_ids:
                unique_ids.append(x)

        with MPRester(mp_api_key, mute_progress_bars=mute_progress_bars) as mpr:
            try:
                query = mpr.materials.summary.search(material_ids=unique_ids, fields=MAT_PROPERTIES)
                logger.info(
                    f"MP query successful, {len(query)} perovskites found through Robocrystallographer and Perovskite Project formulae."
                )
            except Exception as e:
                raise e
        logger.debug("Converting MP query results into DataFrame.")
        perovskites_database = convert_my_query_to_dataframe(query, mute_progress_bars=mute_progress_bars)

        query_ids = list()
        for m in query:
            query_ids.append(m.material_id)

        # Fast cleaning
        logger.debug("Removing NaN (rows)")
        logger.debug(f"size DB before = {len(perovskites_database)}")
        perovskites_database = perovskites_database.dropna(axis=0, how="any", subset=BAND_CRITICAL_FIELD)
        logger.debug(f"size DB after = {len(perovskites_database)}")
        logger.debug("Removing NaN (cols)")
        logger.debug(f"size DB before = {len(perovskites_database)}")
        perovskites_database = perovskites_database.dropna(axis=1, how="all")
        logger.debug(f"size DB after = {len(perovskites_database)}")

        # Filtering
        logger.debug("Removing metallic perovskites.")
        logger.debug(f"size DB before = {len(perovskites_database)}")
        perovskites_database["is_metal"] = perovskites_database["is_metal"].astype(bool)
        filtered_perov_database = perovskites_database[~(perovskites_database["is_metal"])]
        # filtered_perov_database = perovskites_database
        logger.debug(f"size DB after = {len(filtered_perov_database)}")

        query_ids_filtered = filtered_perov_database["material_id"]
        diff = set(query_ids) - set(query_ids_filtered)

        reach_end = False
        while not reach_end:
            for i, q in enumerate(query):
                if q.material_id in diff:
                    query.pop(i)
                    break
            if i == len(query) - 1:
                reach_end = True

        filtered_perov_database.reset_index(drop=True, inplace=True)
        self._perovskites = filtered_perov_database.copy()

        logger.success("Perovskites retrieved successfully.")
        return self._perovskites, query

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
        saving_dir = self.database_directories[stage] / "structures/"
        database = self.get_database(stage)

        # Ensure raw data integrity
        if stage == "raw" and not self._update_raw:
            logger.error("Raw data must be treated as immutable!")
            raise ImmutableRawDataError("Raw data must be treated as immutable!")

        # Clear and recreate directory
        if saving_dir.exists():
            logger.warning(f"Cleaning {saving_dir}")
            sh.rmtree(saving_dir)
        saving_dir.mkdir(parents=True, exist_ok=False)

        # Create a lookup dictionary for material IDs → DataFrame row indices
        material_id_to_index = {mid: idx for idx, mid in enumerate(database["material_id"])}

        logger.debug("MP querying for perovskite structures.")
        mp_api_key = get_mp_api_key()

        with MPRester(mp_api_key, mute_progress_bars=mute_progress_bars) as mpr:
            try:
                materials_mp_query = mpr.materials.summary.search(
                    material_ids=list(material_id_to_index.keys()),
                    fields=["material_id", "structure"],
                )
                logger.info(f"MP query successful, {len(materials_mp_query)} structures found.")
            except Exception as e:
                logger.error(f"MP query failed: {e}")
                raise e

        all_cif_paths = {}  # Store updates in a dict to vectorize DataFrame updates later

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
        database["cif_path"] = database["material_id"].map(all_cif_paths)

        # Save the updated database
        self.save_database(stage)
        logger.info(f"CIF files for stage '{stage}' saved and database updated successfully.")

    def process_database(
        self,
        band_gap_lower: float,
        band_gap_upper: float,
        inplace: bool = True,
        db: pd.DataFrame = None,
        clean_magnetic: bool = True,
    ) -> pd.DataFrame:
        """
        Process the raw perovskite database to the `processed` stage.

        This method filters materials based on their band gap energy range and optionally
        removes metallic and magnetic materials. The processed data can either be saved
        in place or returned as a separate DataFrame.

        Args:
            band_gap_lower (float): Lower bound of the band gap energy range (in eV).
            band_gap_upper (float): Upper bound of the band gap energy range (in eV).
            inplace (bool, optional): If True, updates the "processed" database in-place.
                If False, returns the processed DataFrame. Defaults to True.
            db (pd.DataFrame, optional): The database to process if `inplace` is False. Defaults to None.
            clean_magnetic (bool, optional): If True, removes magnetic materials. Defaults to True.

        Returns:
            (pd.DataFrame): Processed database if `inplace` is False, otherwise None.

        Raises:
            ValueError: If `inplace` is False but no DataFrame (`db`) is provided.

        Logs:
            - INFO: Steps of the processing, including filtering metallic and magnetic materials.
            - ERROR: If `inplace` is False and no input database is provided.
        """

        if not inplace and db is None:
            logger.error("Invalid input: You must input a pd.DataFrame if 'inplace' is set to True.")
            raise ValueError("You must input a pd.DataFrame if 'inplace' is set to True.")

        if inplace:
            raw_db = self.get_database(stage="raw")
        else:
            raw_db = db

        raw_db = raw_db[~(raw_db["is_metal"].astype(bool))]
        logger.info("Removing metallic materials")
        if clean_magnetic:
            temp_db = raw_db[~(raw_db["is_magnetic"].astype(bool))]
            logger.info("Removing magnetic materials")
        else:
            temp_db = raw_db
            logger.info("Keeping magnetic materials")

        logger.info(f"Removing materials with bandgap {band_gap_lower} eV < E_g <= {band_gap_upper} eV")
        processed_db = temp_db[(temp_db["band_gap"] > band_gap_lower) & (temp_db["band_gap"] <= band_gap_upper)]

        processed_db.reset_index(drop=True, inplace=True)

        if inplace:
            self.databases["processed"] = processed_db.copy()
            self.save_database("processed")
        else:
            return processed_db

    def __repr__(self) -> str:
        """
        Text representation of the PerovskiteDatabase instance.
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
        HTML representation of the PerovskiteDatabase instance.
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
