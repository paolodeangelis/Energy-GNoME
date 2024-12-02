# energy_gnome/data/base_dataset.py

"""
Base Dataset Module for Energy Gnome Library.

This module defines the abstract base class `BaseDataSet`, outlining the common
interface and functionalities required for all specialized dataset classes within
the Energy Gnome library. It ensures consistency, enforces a standard structure,
and provides shared utility methods for managing different data processing stages
(raw, processed, final).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from loguru import logger
import pandas as pd

from energy_gnome.config import DATA_DIR


class BaseDatabase(ABC):
    def __init__(self, data_dir: Path = DATA_DIR):
        """
        Initialize the BaseDatabase with a root data directory.

        Sets up the directory structure for storing data across different processing stages
        (raw, processed, final) and initializes placeholders for database paths and data.

        Args:
            data_dir (Path, optional): Root directory path for storing data.
                                       Defaults to Path("data/").
        """
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Define processing stages
        self.processing_stages = ["raw", "processed", "final"]

        # Initialize directories, paths, and databases for each stage
        self.database_directories = {
            stage: self.data_dir / stage for stage in self.processing_stages
        }
        for stage_dir in self.database_directories.values():
            stage_dir.mkdir(parents=True, exist_ok=True)

        self.database_paths = {
            stage: dir_path / "database.json"
            for stage, dir_path in self.database_directories.items()
        }

        self.databases = {stage: pd.DataFrame() for stage in self.processing_stages}
        self._update_raw = False

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
            logger.debug(f"Loaded existing database from {db_path}")
        else:
            logger.warning(f"No existing database found at {db_path}")
        return self.databases[stage]

    def load_all(self):
        """
        Load the databases for all the stages.
        """
        for stage in self.processing_stages:
            self.load_database(stage)

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
