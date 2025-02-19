# energy_gnome/data/base_dataset.py

"""
Base Dataset Module for Energy Gnome Library.

This module defines the abstract base class `BaseDataSet`, outlining the common
interface and functionalities required for all specialized dataset classes within
the Energy Gnome library. It ensures consistency, enforces a standard structure,
and provides shared utility methods for managing different data processing stages
(raw, processed, final).
"""
import os

try:
    import win32file
except ImportError:
    pass

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from loguru import logger
from numpy.random import PCG64, Generator
import pandas as pd
from sklearn.model_selection import train_test_split

from energy_gnome.config import DATA_DIR
from energy_gnome.exception import DatabaseError

from .splitting import random_split


def make_link(source: Path, target: Path):
    if source.exists():
        logger.warning(f"File {source} already exist")
    else:
        os.symlink(source, target)
        logger.info(f"Made link {source} -> {target}")


class BaseDatabase(ABC):
    def __init__(self, name: str, data_dir: Path = DATA_DIR):
        """
        Initialize the BaseDatabase with a root data directory.

        Sets up the directory structure for storing data across different processing stages
        (raw, processed, final) and initializes placeholders for database paths and data.

        Args:
            data_dir (Path, optional): Root directory path for storing data.
                                       Defaults to Path("data/").
        """
        self.name = name

        if isinstance(data_dir, str):
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Define processing stages
        self.processing_stages = ["raw", "processed", "final"]
        self.interim_sets = ["training", "validation", "testing"]

        # Initialize directories, paths, and databases for each stage
        self.database_directories = {
            stage: self.data_dir / stage / self.name for stage in self.processing_stages
        }
        for stage_dir in self.database_directories.values():
            stage_dir.mkdir(parents=True, exist_ok=True)

        self.database_paths = {
            stage: dir_path / "database.json"
            for stage, dir_path in self.database_directories.items()
        }

        self.databases = {stage: pd.DataFrame() for stage in self.processing_stages}
        self.subset = {subset: pd.DataFrame() for subset in self.interim_sets}
        self._update_raw = False
        self.is_specialized = False
        self._set_is_specialized()

    @abstractmethod
    def _set_is_specialized(
        self,
    ):
        pass

    def allow_raw_update(self):
        """
        Allow the 'raw' data to be changed and avoid raising exception.
        """
        self._update_raw = True

    @abstractmethod
    def retrieve_remote(self) -> pd.DataFrame:
        """
        Retrieve data from the Material Project API.
        """
        pass

    @abstractmethod
    def compare_databases(self, new_db: pd.DataFrame, stage: str) -> pd.DataFrame:
        """
        Compare a new database with the existing one to identify differences.

        Args:
            new_db (pd.DataFrame): new database to compare.
            stage (str): The processing stage ('raw', 'processed', 'final').

        Returns:
            pd.DataFrame: Subset of `db_new` containing only new battery IDs.
        """
        pass

    @abstractmethod
    def retrieve_materials(self) -> list[Any]:
        """
        Retrieve material structures from the Material Project API.

        Subclasses must implement this method to fetch material structures.
        The method should interact with the Material Project API, perform necessary
        queries, and return the results as a list of material objects.

        Returns:
            List[Any]: List of retrieved material objects.
        """
        pass

    @abstractmethod
    def save_cif_files(self) -> None:
        """
        Save CIF files for materials and update the database accordingly.
        """
        pass

    @abstractmethod
    def copy_cif_files(self) -> None:
        """
        Copy CIF files for materials and update the database accordingly.
        """
        pass

    def load_database(self, stage: str) -> pd.DataFrame:
        """
        Load the existing database for a specific processing stage.

        Checks for the presence of an existing database file for the given state
        and loads it into a pandas DataFrame. If the database file does not exist,
        logs a warning and returns an empty DataFrame.

        Args:
            stage (str): The processing stage ('raw', 'processed', 'final').

        Returns:
            pd.DataFrame: The loaded database or an empty DataFrame if not found.
        """
        if stage not in self.processing_stages:
            logger.error(f"Invalid stage: {stage}. Must be one of {self.processing_stages}.")
            raise ValueError(f"stage must be one of {self.processing_stages}.")

        db_path = self.database_paths[stage]
        if db_path.exists():
            self.databases[stage] = pd.read_json(db_path)
            if stage == "raw":
                self.databases[stage]["is_specialized"] = self.is_specialized
            logger.debug(f"Loaded existing database from {db_path}")
        else:
            logger.warning(f"Not found at {db_path}")
        return self.databases[stage]

    @abstractmethod
    def _load_interim(self, subset: str) -> None:
        pass

    def load_regressor_data(self, subset: str = "training"):
        return self._load_interim(subset=subset, model_type="regressor")

    def load_classifier_data(self, subset: str = "training"):
        return self._load_interim(subset=subset, model_type="classifier")

    def load_all(self):
        """
        Load the databases for all the stages.
        """
        for stage in self.processing_stages:
            self.load_database(stage)
        for subset in self.interim_sets:
            self.load_regressor_data(subset)
            self.load_classifier_data(subset)

    def save_database(self, stage: str) -> None:
        """
        Save the current state of the database to a JSON file.

        Serializes the current database DataFrame and saves it to a JSON file
        at the designated path. Logs the success of the operation or any errors encountered.

        Args:
            stage (str): The processing stage ('raw', 'processed', 'final').

        Raises:
            IOError: If there is an issue writing the DataFrame to the file.
        """
        if stage not in self.processing_stages:
            logger.error(f"Invalid stage: {stage}. Must be one of {self.processing_stages}.")
            raise ValueError(f"stage must be one of {self.processing_stages}.")

        db_path = self.database_paths[stage]
        if os.path.exists(db_path):
            os.unlink(db_path)
        try:
            self.databases[stage].to_json(db_path)
            logger.info(f"Database saved to {db_path}")
        except Exception as e:
            logger.error(f"Failed to save database to {db_path}: {e}")
            raise OSError(f"Failed to save database to {db_path}: {e}") from e

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
            differences (pd.DataFrame): The database with the items that are new or updated.
            stage (str): The processing stage ('raw', 'processed', 'final').

        Raises:
            IOError: If there is an issue writing to the backup or changelog files.
        """
        if stage not in self.processing_stages:
            logger.error(f"Invalid stage: {stage}. Must be one of {self.processing_stages}.")
            raise ValueError(f"stage must be one of {self.processing_stages}.")

        from datetime import datetime

        backup_path = self.database_directories[stage] / "old_database.json"
        try:
            old_db.to_json(backup_path)
            logger.debug(f"Old database backed up to {backup_path}")
        except Exception as e:
            logger.error(f"Failed to backup old database to {backup_path}: {e}")
            raise OSError(f"Failed to backup old database to {backup_path}: {e}") from e

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        changelog_path = self.data_dir / "changelog.txt"
        changelog_entries = [
            f"Change Log - {timestamp}\n",
            f"{'ID':<15}{'Formula':<30}{'Last Updated (MP)':<20}\n",
            "-" * 65 + "\n",
        ]

        for identifier in differences:
            row = new_db.loc[new_db.index == identifier]
            if not row.empty:
                formula = row["formula"].values[0]
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

    def get_database(self, stage, subset=None):
        if stage not in self.processing_stages + ["interim"]:
            logger.error(
                f"Invalid stage: {stage}. Must be one of {self.processing_stages + ['interim']}."
            )
            raise ValueError(f"stage must be one of {self.processing_stages + ['interim']}.")
        if stage in self.processing_stages:
            out_db = self.databases[stage]
        elif stage in ["interim"] and subset is not None:
            out_db = self.subset[subset]
        else:
            raise ValueError(f"subset must be one of {self.interim_sets}.")

        if len(out_db) == 0:
            logger.warning("Empty database found.")
        return out_db

    def build_reduced_database(
        self,
        size: int,
        new_name: str,
        stage: str,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Build the reduced MP database.

        Args:
            database (pd.DataFrame): The database from which to pick random entries.
            size (int): The size (number of entries) of the reduced database.
            seed (int): The random seed used for creating reproducible databases. Defaults to 42.

        Returns:
            pd.DataFrame: The filtered generic database.
        """
        new_database = self.__class__(data_dir=self.data_dir, name=new_name)
        if size == 0:
            logger.error("You are creating an empty database.")
            raise ValueError("You are creating an empty database.")
        # copy database
        for stages, db in self.databases.items():
            if len(self.databases[stages]) != 0:
                new_database.databases[stages] = db.copy()
                make_link(self.database_paths[stages], new_database.database_paths[stages])
        for subset, db in self.subset.items():
            if len(self.subset[subset]) != 0:
                new_database.subset[subset] = db.copy()
                subset_path_father = self.data_dir / "interim" / self.name / (subset + "_db.json")
                subset_path_son = (
                    new_database.data_dir / "interim" / new_database.name / (subset + "_db.json")
                )
                if not subset_path_son.parent.exists():
                    subset_path_son.parent.mkdir(parents=True, exist_ok=True)
                make_link(subset_path_father, subset_path_son)

        database = new_database.databases[stage].copy()
        n_all = len(database)
        rng = Generator(PCG64(seed))
        row_i_all = rng.choice(n_all, size, replace=False)
        new_database.databases[stage] = database.iloc[row_i_all, :].reset_index(drop=True)

        new_database.save_database(stage)

        return new_database

    def save_split_db(self, database_dict: dict, model_type: str = "regressor"):
        """
        Save the split databases.
        """
        db_path = self.data_dir / "interim" / self.name / model_type
        # if not db_path.exists():
        db_path.mkdir(parents=True, exist_ok=True)

        split_paths = dict(
            train=db_path / "training_db.json",
            valid=db_path / "validation_db.json",
            test=db_path / "testing_db.json",
        )

        for db_name, db_path in split_paths.items():
            if database_dict[db_name].empty:
                if db_name == "valid":
                    logger.warning("No validation database provided.")
                    continue
                elif db_name == "test":
                    logger.warning("No testing database provided.")
                    continue
                else:
                    raise DatabaseError(f"The {db_name} is empty, check the splitting.")
            else:
                database_dict[db_name].to_json(db_path)
                logger.info(f"{db_name} database saved to {db_path}")

    def split_regressor(
        self,
        target_property: str,
        valid_size: float = 0.2,
        test_size: float = 0.05,
        seed: int = 42,
        balance_composition: bool = True,
        save_split: bool = False,
    ):
        if balance_composition:
            db_dict = random_split(
                self.get_database("processed"),
                target_property,
                valid_size=valid_size,
                test_size=test_size,
                seed=seed,
            )
        else:
            dev_size = valid_size + test_size
            if abs(dev_size - test_size) < 1e-8:
                train_, test_ = train_test_split(
                    self.get_database("processed"), test_size=test_size, random_state=seed
                )
                valid_ = pd.DataFrame()
            elif abs(dev_size - valid_size) < 1e-8:
                train_, valid_ = train_test_split(
                    self.get_database("processed"), test_size=valid_size, random_state=seed
                )
                test_ = pd.DataFrame()
            else:
                train_, dev_ = train_test_split(
                    self.get_database("processed"),
                    test_size=valid_size + test_size,
                    random_state=seed,
                )
                valid_, test_ = train_test_split(
                    dev_, test_size=test_size / (valid_size + test_size), random_state=seed
                )

            db_dict = {"train": train_, "valid": valid_, "test": test_}

        if save_split:
            self.save_split_db(db_dict, "regressor")

        self.subset = db_dict

    def split_classifier(
        self,
        test_size: float = 0.2,
        seed: int = 42,
        balance_composition: bool = False,
        save_split: bool = False,
    ):
        target_property = "is_specialized"
        if balance_composition:
            db_dict = random_split(
                self.get_database("processed"),
                target_property,
                valid_size=0,
                test_size=test_size,
                seed=seed,
            )
        else:
            train_, test_ = train_test_split(
                self.get_database("processed"), test_size=test_size, random_state=seed
            )
            db_dict = {"train": train_, "test": test_}

        if save_split:
            self.save_split_db(db_dict, "classifier")

        self.subset = db_dict

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
            "</div>"
        )
        return html
