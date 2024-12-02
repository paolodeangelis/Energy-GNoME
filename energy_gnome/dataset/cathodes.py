from datetime import datetime
from pathlib import Path
import shutil as sh
from typing import Any

from mp_api.client import MPRester
import pandas as pd
from tqdm import tqdm

from energy_gnome.config import DATA_DIR, DOI_ARTICLE, RAW_DATA_DIR  # noqa:401
from energy_gnome.dataset.base_dataset import BaseDatabase
from energy_gnome.exception import ImmutableRawDataError, MissingData
from energy_gnome.utils.logger_config import logger
from energy_gnome.utils.mp_api_utils import (
    convert_my_query_to_dataframe,
    get_material_by_id,
    get_mp_api_key,
)

# Paths
CATHODES_RAW_DATA_DIR = RAW_DATA_DIR / "cathodes"


BAT_FIELDS = [
    "battery_id",
    "thermo_type",
    "battery_formula",
    "working_ion",
    "num_steps",
    "max_voltage_step",
    "last_updated",
    "framework",
    "framework_formula",
    "elements",
    "nelements",
    "warnings",
    "formula_charge",
    "formula_discharge",
    "max_delta_volume",
    "average_voltage",
    "capacity_grav",
    "capacity_vol",
    "energy_grav",
    "energy_vol",
    "fracA_charge",
    "fracA_discharge",
    "stability_charge",
    "stability_discharge",
    "id_charge",
    "id_discharge",
    "adj_pairs",
    "material_ids",
]

MAT_PROPERTIES = {
    "volume": "float64",
    "density": "float64",
    "energy_per_atom": "float64",
    "formation_energy_per_atom": "float64",
    "formula_pretty": "str",
    "material_id": "str",
    "is_stable": "bool",
}


class CathodeDatabase(BaseDatabase):
    def __init__(
        self, data_dir: Path = DATA_DIR, working_ion: str = "Li", battery_type: str = "insertion"
    ):
        """
        Initialize the CathodeDatabase with a root data directory and processing stage.

        Sets up the directory structure for storing data across different processing stages
        (`raw/`, `processed/`, `final/`) and initializes placeholders for database paths and data.

        Args:
            data_dir (Path, optional): Root directory path for storing data.
                                       Defaults to DATA_DIR from config.
            working_ion (str, optional): The working ion used in the dataset (e.g., 'Li').
                                         Defaults to "Li".
            battery_type (str, optional): The type of battery type (e.g., 'insertion', 'conversion').
                                          Defaults to "insertion".

        Raises:
            NotImplementedError: If the specified processing stage is not supported.
            ImmutableRawDataError: If attempting to set an unsupported processing stage.
        """
        super().__init__(data_dir=data_dir)
        self.working_ion = working_ion

        if battery_type == "insertion":
            self.battery_type = battery_type
        elif battery_type == "conversion":
            logger.error("`conversion` battery type is not yet implemented in Material Project.")
            raise NotImplementedError(
                "`conversion` battery type is not yet present in Material Project."
            )
        else:
            logger.error(
                f"Invalid battery type: {battery_type}. Must be 'insertion' or 'conversion'."
            )
            raise ValueError(
                "`battery_type` can be only `insertion` or `conversion` (not yet implemented)"
            )

        # Initialize directories, paths, and databases for each stage
        self.database_directories = {
            stage: self.data_dir / stage / "cathodes" / battery_type / working_ion
            for stage in self.processing_stages
        }
        for stage_dir in self.database_directories.values():
            stage_dir.mkdir(parents=True, exist_ok=True)

        self.database_paths = {
            stage: dir_path / "database.json"
            for stage, dir_path in self.database_directories.items()
        }

        self.databases = {stage: pd.DataFrame() for stage in self.processing_stages}
        self._battery_models = pd.DataFrame()

    def retrieve_remote(self, mute_progress_bars: bool = True) -> pd.DataFrame:
        """
        Retrieve models from the Material Project API.

        Wrapper method to call `retrieve_models`.

        Args:
            mute_progress_bars (bool, optional):
                If `True`, mutes the Material Project API progress bars.
                Defaults to `True`.

        Returns:
            pd.DataFrame: DataFrame containing the retrieved models.
        """
        return self.retrieve_models(mute_progress_bars=mute_progress_bars)

    def retrieve_models(self, mute_progress_bars: bool = True) -> pd.DataFrame:
        """
        Retrieve battery models from the Materials Project API.

        Connects to the Material Project API using MPRester, queries for materials
        based on the working ion and processing stage, and retrieves the specified fields.
        Cleans the data by removing entries with missing critical identifiers.

        Args:
            mute_progress_bars (bool, optional):
                If `True`, mutes the Material Project API progress bars.
                Defaults to `True`.

        Returns:
            pd.DataFrame: DataFrame containing the retrieved and cleaned models.

        Raises:
            Exception: If the API query fails.
        """
        mp_api_key = get_mp_api_key()
        logger.debug("MP querying for insertion battery models.")

        with MPRester(mp_api_key, mute_progress_bars=mute_progress_bars) as mpr:
            try:
                query = mpr.materials.insertion_electrodes.search(
                    working_ion=self.working_ion, fields=BAT_FIELDS
                )
                logger.info(
                    f"MP query successful, {len(query)} {self.working_ion}-ion batteries found."
                )
            except Exception as e:
                raise e
        logger.debug("Converting MP query results into DataFrame.")
        battery_models_database = convert_my_query_to_dataframe(
            query, mute_progress_bars=mute_progress_bars
        )

        # Fast cleaning
        logger.debug("Removing NaN")
        battery_models_database = battery_models_database.dropna(
            axis=0, how="any", subset=["id_charge", "id_discharge"]
        )
        battery_models_database = battery_models_database.dropna(axis=1, how="all")
        self._battery_models = battery_models_database
        logger.success(f"{self.working_ion}-ion batteries model retrieved successfully.")
        return self._battery_models

    def compare_databases(self, new_db: pd.DataFrame, stage: str) -> pd.DataFrame:
        """
        Compare two databases and identify new entry IDs.

        Args:
            new_db (pd.DataFrame): New database to compare.
            stage (str): Processing stage ("raw", "processed", "final").

        Returns:
            pd.DataFrame: Subset of `new_db` containing only new entry IDs.
        """
        old_db = self.load_database(stage=stage)
        if not old_db.empty:
            new_ids_set = set(new_db["battery_id"])
            old_ids_set = set(old_db["battery_id"])
            new_ids_only = new_ids_set - old_ids_set
            logger.debug(f"Found {len(new_ids_only)} new battery IDs in the new database.")
            return new_db[new_db["battery_id"].isin(new_ids_only)]
        else:
            logger.warning("Nothing to compare here...")
            return new_db

    def backup_and_changelog(
        self,
        old_db: pd.DataFrame,
        new_db: pd.DataFrame,
        differences: pd.Series,
        stage: str,
    ) -> None:
        """
        Backup the old database and update the changelog with identified differences.

        Creates a backup of the existing database and appends a changelog entry detailing
        the differences between the old and new databases. The changelog includes
        information such as entry identifiers, formulas, and last updated timestamps.

        Args:
            old_db (pd.DataFrame): The existing database before updates.
            new_db (pd.DataFrame): The new database containing updates.
            differences (pd.Series): Series of identifiers that are new or updated.
            stage (str): The processing stage ('raw', 'processed', 'final').
        """
        if stage not in self.processing_stages:
            logger.error(f"Invalid stage: {stage}. Must be one of {self.processing_stages}.")
            raise ValueError(f"stage must be one of {self.processing_stages}.")

        backup_path = self.database_directories[stage] / "old_database.json"
        try:
            old_db.to_json(backup_path)
            logger.debug(f"Old database backed up to {backup_path}")
        except Exception as e:
            logger.error(f"Failed to backup old database to {backup_path}: {e}")
            raise OSError(f"Failed to backup old database to {backup_path}: {e}") from e

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        changelog_path = self.database_directories[stage] / "changelog.txt"
        changelog_entries = [
            f"= Change Log - {timestamp} ".ljust(70, "=") + "\n",
            "Difference old_database.json VS database.json\n",
            f"{'ID':<15}{'Formula':<30}{'Last Updated (MP)':<25}\n",
            "-" * 70 + "\n",
        ]
        # Tailoring respect father class
        for identifier in differences["battery_id"]:
            row = new_db.loc[new_db["battery_id"] == identifier]
            if not row.empty:
                formula = row["battery_formula"].values[0]
                last_updated = row["last_updated"].values[0]
            else:
                formula = "N/A"
                last_updated = "N/A"
            changelog_entries.append(f"{identifier:<15}{formula:<30}{last_updated:<20}\n")

        try:
            with open(changelog_path, "a") as file:
                file.writelines(changelog_entries)
            logger.debug(f"Changelog updated at {changelog_path} with {len(differences)} changes.")
        except Exception as e:
            logger.error(f"Failed to update changelog at {changelog_path}: {e}")
            raise OSError(f"Failed to update changelog at {changelog_path}: {e}") from e

    def compare_and_update(self, new_db: pd.DataFrame, stage: str) -> pd.DataFrame:
        """
        Compare and update the database with new entries.

        Identifies new entries and updates the database accordingly. Ensures that raw data
        remains immutable by preventing updates unless explicitly allowed.

        Args:
            new_db (pd.DataFrame): New database to compare.
            stage (str): Processing stage ("raw", "processed", "final").

        Returns:
            pd.DataFrame: Updated database containing new entries.

        Raises:
            ImmutableRawDataError: If attempting to modify immutable raw data.
        """
        old_db = self.load_database(stage=stage)
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
                    logger.info(
                        "Be careful you are changing the raw data which must be treated as immutable!"
                    )
                logger.info(
                    f"Updating the {stage} data and saving it in {self.database_paths[stage]}."
                )
                self.backup_and_changelog(
                    old_db,
                    new_db,
                    db_diff,
                    stage,
                )
                self.databases[stage] = new_db
                self.save_database(stage)
        else:
            logger.info("No new items found. No update required.")

    def retrieve_materials(
        self, stage: str, charge_state: str, mute_progress_bars: bool = True
    ) -> list[Any]:
        """
        Retrieve material structures from the Material Project API.

        Fetches material structures based on the processing stage and charge state.

        Args:
            stage (str): Processing stage ('raw', 'processed', 'final').
            charge_state (str): Cathode charge state ('charge', 'discharge').
            mute_progress_bars (bool, optional): Disable progress bar if True. Defaults to True.

        Returns:
            List[Any]: List of retrieved material objects.

        Raises:
            ValueError: If the charge_state is invalid.
            MissingData: If the required data is missing in the database.
        """
        if charge_state not in ["charge", "discharge"]:
            logger.error(f"Invalid charge_state: {charge_state}. Must be 'charge' or 'discharge'.")
            raise ValueError("charge_state must be 'charge' or 'discharge'.")

        material_ids = self.databases[stage][f"id_{charge_state}"].tolist()
        if not material_ids:
            logger.warning(
                f"No material IDs found for stage '{stage}' and charge_state '{charge_state}'."
            )
            raise MissingData(
                f"No material IDs found for stage '{stage}' and charge_state '{charge_state}'."
            )

        logger.debug(
            f"Retrieving materials for stage '{stage}' and charge_state '{charge_state}'."
        )
        query = get_material_by_id(
            material_ids,
            mute_progress_bars=mute_progress_bars,
        )
        return query

    def _add_materials_properties_columns(self, stage: str, charge_state: str) -> pd.DataFrame:
        """
        Add material properties columns to the database for a given cathode state.

        Args:
            stage (str): Processing stage ('raw', 'processed', 'final').
            charge_state (str): Cathode charge state ('charge', 'discharge').

        Returns:
            pd.DataFrame: Updated database with material properties columns.

        Raises:
            ImmutableRawDataError: If attempting to modify immutable raw data.
        """
        if charge_state not in ["charge", "discharge"]:
            logger.error(f"Invalid charge_state: {charge_state}. Must be 'charge' or 'discharge'.")
            raise ValueError("charge_state must be 'charge' or 'discharge'.")

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
                logger.info(
                    "Be careful you are changing the raw data which must be treated as immutable!"
                )
            logger.debug(
                f"Adding material properties to {stage} data for cathode state: {charge_state}"
            )
            for property_name, dtype in MAT_PROPERTIES.items():
                column_name = f"{charge_state}_{property_name}"
                if column_name not in self.databases[stage].columns:
                    logger.debug(f"Adding missing column: {column_name} with dtype {dtype}")
                    self.databases[stage][column_name] = pd.Series(dtype=dtype)

    def add_material_properties(
        self,
        stage: str,
        materials_mp_query: list,
        charge_state: str,
        mute_progress_bars: bool = True,
    ) -> pd.DataFrame:
        """
        Add material properties to the database from Material Project query results.

        Saves CIF files for each material in the query and updates the database with file paths and properties.

        Args:
            stage (str): Processing stage ('raw', 'processed', 'final').
            materials_mp_query (List[Any]): List of material query results.
            charge_state (str): The state of the cathode ('charge' or 'discharge').
            mute_progress_bars (bool, optional): Disable progress bar if True. Defaults to True.

        Returns:
            pd.DataFrame: Updated database with material properties.

        Raises:
            ImmutableRawDataError: If attempting to modify immutable raw data.
            KeyError: If a material ID is not found in the database.
        """
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
                logger.info(
                    "Be careful you are changing the raw data which must be treated as immutable!"
                )
            logger.debug(
                f"Adding material properties to {stage} data for cathode state: {charge_state}"
            )

            # Ensure necessary columns are present
            self._add_materials_properties_columns(stage, self.databases[stage], charge_state)

            for material in tqdm(
                materials_mp_query,
                desc=f"Adding {charge_state} cathodes properties",
                disable=mute_progress_bars,
            ):
                try:
                    # Locate the row in the database corresponding to the material ID
                    i_row = (
                        self.databases[stage]
                        .index[self.databases[stage][f"id_{charge_state}"] == material.material_id]
                        .tolist()[0]
                    )

                    # Assign material properties to the database
                    for property_name in MAT_PROPERTIES.keys():
                        self.databases[stage].at[i_row, f"{charge_state}_{property_name}"] = (
                            getattr(material, property_name, None)
                        )
                except IndexError:
                    logger.error(f"Material ID {material.material_id} not found in the database.")
                    raise MissingData(
                        f"Material ID {material.material_id} not found in the database."
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to add properties for Material ID {material.material_id}: {e}"
                    )
                    raise e

        logger.info(f"Material properties for '{charge_state}' cathodes added successfully.")

    def save_cif_files(
        self,
        stage: str,
        materials_mp_query: list,
        charge_state: str,
        mute_progress_bars: bool = True,
    ) -> None:
        """
        Save CIF files for materials and update the database accordingly.

        Manages the saving of CIF files for each material and updates the database with
        the file paths and relevant properties. Ensures that raw data remains immutable.

        Args:
            stage (str): Processing stage ('raw', 'processed', 'final').
            materials_mp_query (List[Any]): List of material query results.
            charge_state (str): The charge state of the cathode ('charge' or 'discharge').
            mute_progress_bars (bool, optional): Disable progress bar if True. Defaults to True.

        Raises:
            ImmutableRawDataError: If attempting to modify immutable raw data.
        """

        saving_dir = self.database_directories[stage] / charge_state

        if stage == "raw" and not self._update_raw:
            logger.error("Raw data must be treated as immutable!")
            logger.error(
                "It's okay to read and copy raw data to manipulate it into new outputs, but never okay to change it in place."
            )
            raise ImmutableRawDataError(
                "Raw data must be treated as immutable!\n"
                "It's okay to read and copy raw data to manipulate it into new outputs, but never okay to change it in place."
            )
        elif stage == "raw" and saving_dir.exists():
            logger.info(
                "Be careful you are changing the raw data which must be treated as immutable!"
            )

        # Clean the saving directory if it exists
        if saving_dir.exists():
            logger.warning(f"Cleaning the content in {saving_dir}")
            sh.rmtree(saving_dir)

        # Create the saving directory
        saving_dir.mkdir(parents=True, exist_ok=False)
        self.databases[stage][f"{charge_state}_path"] = pd.Series(dtype=str)

        # Save CIF files and update database paths
        for material in tqdm(
            materials_mp_query,
            desc=f"Saving {charge_state} cathodes",
            disable=mute_progress_bars,
        ):
            try:
                # Locate the row in the database corresponding to the material ID
                i_row = (
                    self.databases[stage]
                    .index[self.databases[stage][f"id_{charge_state}"] == material.material_id]
                    .tolist()[0]
                )

                # Define the CIF file path
                cif_path = saving_dir / f"{material.material_id}.cif"

                # Save the CIF file
                material.structure.to(filename=str(cif_path))

                # Update the database with the CIF file path
                self.databases[stage].at[i_row, f"{charge_state}_path"] = str(cif_path)

            except IndexError:
                logger.error(f"Material ID {material.material_id} not found in the database.")
                raise MissingData(f"Material ID {material.material_id} not found in the database.")
            except Exception as e:
                logger.error(f"Failed to save CIF for Material ID {material.material_id}: {e}")
                raise OSError(
                    f"Failed to save CIF for Material ID {material.material_id}: {e}"
                ) from e

        # Save the updated database
        self.save_database(stage)
        logger.info(f"CIF files for stage '{stage}' saved and database updated successfully.")

    def copy_cif_files(
        self,
        stage: str,
        charge_state: str,
        mute_progress_bars: bool = True,
    ) -> None:
        """
        Copy CIF files from the raw stage to another processing stage.

        Copies CIF files corresponding to the specified cathode state from the 'raw'
        processing stage to the target stage. Updates the database with the new file paths.

        Args:
            stage (str): Target processing stage ('processed', 'final').
            charge_state (str): The charge state of the cathode ('charge' or 'discharge').
            mute_progress_bars (bool, optional): Disable progress bar if True. Defaults to True.

        Raises:
            ValueError: If the target stage is 'raw'.
            MissingData: If the source CIF directory does not exist or is empty.
        """
        if stage == "raw":
            logger.error("Stage argument cannot be 'raw'.")
            logger.error("You can only copy from 'raw' to other stages, not to 'raw' itself.")
            raise ValueError("Stage argument cannot be 'raw'.")

        source_dir = self.database_directories["raw"] / charge_state
        saving_dir = self.database_directories[stage] / charge_state

        # Clean the saving directory if it exists
        if saving_dir.exists():
            logger.warning(f"Cleaning the content in {saving_dir}")
            sh.rmtree(saving_dir)

        # Check if source CIF directory exists and is not empty
        if not source_dir.exists() or not any(source_dir.iterdir()):
            logger.warning(
                f"The raw CIF directory does not exist or is empty. Check: {source_dir}"
            )
            raise MissingData(
                f"The raw CIF directory does not exist or is empty. Check: {source_dir}"
            )

        # Create the saving directory
        saving_dir.mkdir(parents=True, exist_ok=False)
        self.databases[stage][f"{charge_state}_path"] = pd.Series(dtype=str)

        # Copy CIF files and update database paths
        for material_id in tqdm(
            self.databases[stage][f"id_{charge_state}"],
            desc=f"Copying {charge_state} cathodes ('raw' -> '{stage}')",
            disable=mute_progress_bars,
        ):
            try:
                # Locate the row in the database corresponding to the material ID
                i_row = (
                    self.databases[stage]
                    .index[self.databases[stage][f"id_{charge_state}"] == material_id]
                    .tolist()[0]
                )

                # Define source and destination CIF file paths
                source_cif_path = source_dir / f"{material_id}.cif"
                cif_path = saving_dir / f"{material_id}.cif"

                # Copy the CIF file
                sh.copyfile(source_cif_path, cif_path)

                # Update the database with the new CIF file path
                self.databases[stage].at[i_row, f"{charge_state}_path"] = str(cif_path)

            except IndexError:
                logger.error(f"Material ID {material_id} not found in the database.")
                raise MissingData(f"Material ID {material_id} not found in the database.")
            except Exception as e:
                logger.error(f"Failed to copy CIF for Material ID {material_id}: {e}")
                raise OSError(f"Failed to copy CIF for Material ID {material_id}: {e}") from e

        # Save the updated database
        self.save_database(stage)
        logger.info(f"CIF files copied to stage '{stage}' and database updated successfully.")

    def __repr__(self) -> str:
        """
        Text representation of the CathodeDatabase instance.
        Used for print() and str() calls.

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

        header = (
            "|" + "|".join(f" {col:<{widths[i]}}" for i, col in enumerate(info_df.columns)) + "|"
        )
        lines.append(header)
        lines.append(separator)

        # Add data rows
        for _, row in info_df.iterrows():
            line = "|" + "|".join(f" {str(val):<{widths[i]}}" for i, val in enumerate(row)) + "|"
            lines.append(line)

        # Add bottom separator
        lines.append(separator)

        # Add additional info
        lines.append(f"\nWorking Ion: {self.working_ion}")
        lines.append(f"Battery Type: {self.battery_type}")

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """
        HTML representation of the CathodeDatabase instance.
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
            f'<th style="padding: 12px 15px; text-align: left;">{col}</th>'
            for col in info_df.columns
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
            f"Working Ion: {self.working_ion}<br>"
            f"Battery Type: {self.battery_type}"
            "</div>"
            "</div>"
        )
        return html
