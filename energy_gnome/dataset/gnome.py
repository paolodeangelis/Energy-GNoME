from datetime import datetime
from functools import partial
import json
from multiprocessing import Pool, get_context
import os
from pathlib import Path
import shutil as sh
import subprocess
from typing import Any
import zipfile

import numpy as np
import pandas as pd
from tqdm import tqdm

from energy_gnome.config import DATA_DIR  # noqa:401
from energy_gnome.dataset.base_dataset import BaseDatabase
from energy_gnome.exception import ImmutableRawDataError, MissingData
from energy_gnome.utils.logger_config import logger

# Paths
GNoME_DATA_DIR = DATA_DIR / "external" / "gdm_materials_discovery" / "gnome_data"


class GNoMEDatabase(BaseDatabase):
    def __init__(self, name: str = "gnome", data_dir: Path | str = DATA_DIR):
        """
        Initialize the GNoMEDatabase with a root data directory and processing stage.

        Sets up the directory structure for storing data across different processing stages
        (`raw/`, `processed/`, `final/`) and initializes placeholders for database paths and data.

        Args:
            data_dir (Path, optional): Root directory path for storing data.
                                       Defaults to DATA_DIR from config.

        Raises:
            NotImplementedError: If the specified processing stage is not supported.
            ImmutableRawDataError: If attempting to set an unsupported processing stage.
        """
        super().__init__(data_dir=data_dir, name=name)

        # Initialize directories, paths, and databases for each stage
        self.database_directories = {
            stage: self.data_dir / stage / self.name for stage in self.processing_stages
        }
        # Force single directory for raw database of GNoMEDatabase
        self.database_directories["raw"] = self.data_dir / "raw" / "gnome"

        for stage_dir in self.database_directories.values():
            stage_dir.mkdir(parents=True, exist_ok=True)

        self.database_paths = {
            stage: dir_path / "database.json"
            for stage, dir_path in self.database_directories.items()
        }

        self.databases = {stage: pd.DataFrame() for stage in self.processing_stages}
        self.subset = {subset: pd.DataFrame() for subset in self.interim_sets}
        self._gnome = pd.DataFrame()

    def _set_is_specialized(
        self,
    ):
        pass

    def retrieve_remote(self) -> pd.DataFrame:
        pass

    def _load_interim(self, subset: str) -> None:
        pass

    def load_all(self):
        """
        Load the databases for all the stages.
        """
        for stage in self.processing_stages:
            self.load_database(stage)

    def retrieve_materials(self) -> pd.DataFrame:
        """
        Retrieve material structures.

        Subclasses must implement this method to fetch material structures.

        Returns:
            List[Any]: List of retrieved material objects.
        """
        csv_db = pd.read_csv(GNoME_DATA_DIR / "stable_materials_summary.csv")

        return csv_db

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
            new_ids_set = set(new_db["MaterialId"])
            old_ids_set = set(old_db["MaterialId"])
            new_ids_only = new_ids_set - old_ids_set
            logger.debug(f"Found {len(new_ids_only)} new material IDs in the new database.")
            return new_db[new_db["MaterialId"].isin(new_ids_only)]
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
        """
        if stage not in self.processing_stages:
            logger.error(f"Invalid stage: {stage}. Must be one of {self.processing_stages}.")
            raise ValueError(f"stage must be one of {self.processing_stages}.")

        # Backup the old database
        backup_path = self.database_directories[stage] / "old_database.json"
        try:
            old_db.to_json(backup_path)
            logger.debug(f"Old database backed up to {backup_path}")
        except Exception as e:
            logger.error(f"Failed to backup old database to {backup_path}: {e}")
            raise OSError(f"Failed to backup old database to {backup_path}: {e}") from e

        # Prepare changelog
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        changelog_path = self.database_directories[stage] / "changelog.txt"

        header = (
            f"= Change Log - {timestamp} ".ljust(70, "=") + "\n"
            "Difference old_database.json VS database.json\n"
            f"{'ID':<15}{'Formula':<30}\n" + "-" * 70 + "\n"
        )

        # Set index for faster lookup
        new_db_indexed = new_db.set_index("MaterialId")

        # Process differences efficiently
        changes = [
            f"{identifier:<15}{new_db_indexed.at[identifier, 'Reduced Formula'] if identifier in new_db_indexed.index else 'N/A':<30}"
            for identifier in differences["MaterialId"]
        ]

        try:
            with open(changelog_path, "a") as file:
                file.write(header + "".join(changes))
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

    def save_cif_files(self) -> None:
        """
        Save CIF files for materials and update the database accordingly.
        Uses OS-native unzippers for maximum speed.
        """
        zip_path = GNoME_DATA_DIR / "by_id.zip"
        output_path = self.database_directories["raw"] / "structures"
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("Unzipping structures from")
        logger.info(f"{zip_path}")
        logger.info("to")
        logger.info(f"{output_path}")
        logger.info("using native OS unzip...")

        # OS-native unzip
        if os.name == "nt":  # Windows
            unzip_command = ["tar", "-xf", zip_path, "-C", output_path]
        else:  # Linux/macOS
            unzip_command = ["unzip", "-o", zip_path, "-d", output_path]

        try:
            subprocess.run(unzip_command, check=True, capture_output=True, text=True)
            logger.info("Extraction complete!")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error unzipping file: {e.stderr}")
            return

        # Check if "by_id" subfolder exists
        extracted_dir = output_path / "by_id"
        if extracted_dir.exists() and extracted_dir.is_dir():
            logger.info("Flattening extracted files by moving contents from")
            logger.info(f"{extracted_dir}")
            logger.info("to")
            logger.info(f"{output_path}")

            if os.name == "nt":
                # Windows - use native move command
                move_command = ["move", str(extracted_dir) + "\\*", str(output_path)]
            else:
                # Linux/macOS - use `rsync` or `mv` for efficiency
                move_command = ["mv", str(extracted_dir) + "/*", str(output_path)]

            try:
                subprocess.run(
                    move_command, check=True, shell=True, capture_output=True, text=True
                )
                extracted_dir.rmdir()  # Remove the now-empty folder
            except subprocess.CalledProcessError as e:
                logger.error(f"Error moving files: {e.stderr}")
                return

        # Update Database
        df = self.get_database("raw")
        df["cif_path"] = (
            df["MaterialId"].astype(str).apply(lambda x: (output_path / f"{x}.CIF").as_posix())
        )

        self.save_database("raw")
        logger.info("CIF files saved and database updated successfully.")

    def copy_cif_files(self):
        pass
