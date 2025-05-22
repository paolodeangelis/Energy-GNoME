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

        This constructor sets up the directory structure for storing data across different
        processing stages (`raw`, `processed`, `final`). It also initializes placeholders
        for the database paths and data specific to the `GNoMEDatabase` class.

        Args:
            name (str, optional): The name of the database. Defaults to "gnome".
            data_dir (Path or str, optional): Root directory path for storing data. Defaults
                to `DATA_DIR` from the configuration.

        Raises:
            NotImplementedError: If the specified processing stage is not supported.
            ImmutableRawDataError: If attempting to set an unsupported processing stage.
        """
        super().__init__(name=name, data_dir=data_dir)

        # Force single directory for raw database of GNoMEDatabase
        self.database_directories["raw"] = self.data_dir / "raw" / "gnome"

        self._gnome = pd.DataFrame()
        self._is_specialized = False

    def retrieve_materials(self) -> pd.DataFrame:
        """
        TBD (after implementing the fetch routine)
        """
        csv_db = pd.read_csv(GNoME_DATA_DIR / "stable_materials_summary.csv", index_col=0)
        csv_db = csv_db.rename(columns={"MaterialId": "material_id"})

        return csv_db

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
            logger.error(f"Error unzipping CIF file: {e.stderr}")
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
                move_command = ["robocopy", str(extracted_dir), str(output_path), "/move", "/e"]
            else:
                # Linux/macOS - use `rsync` or `mv` for efficiency
                move_command = ["mv", str(extracted_dir) + "/*", str(output_path)]

            try:
                subprocess.run(move_command, check=True, shell=True, capture_output=True, text=True)
                extracted_dir.rmdir()  # Remove the now-empty folder
            except subprocess.CalledProcessError as e:
                logger.error(f"Error moving files: {e.stderr}")
                return

        # Update Database
        df = self.get_database("raw")
        df["cif_path"] = df["material_id"].astype(str).apply(lambda x: (output_path / f"{x}.CIF").as_posix())

        self.save_database("raw")
        logger.info("CIF files saved and database updated successfully.")

    def copy_cif_files(self):
        pass

    '''
    def filter_by_elements(
            self,
            include: list[str] = None,
            exclude: list[str] = None
        ) -> pd.DataFrame:
        """
        Filters the database entries based on the presence or absence of specified chemical elements.

        Args:
            include (list[str], optional): A list of chemical elements that must be present in the composition.
                                        If None, no filtering is applied based on inclusion.
            exclude (list[str], optional): A list of chemical elements that must not be present in the composition.
                                        If None, no filtering is applied based on exclusion.

        Returns:
            (pd.DataFrame): A filtered DataFrame containing only the entries that match the inclusion/exclusion
                            criteria.

        Raises:
            ValueError: If both `include` and `exclude` lists are empty.

        Notes:
            - If both `include` and `exclude` are provided, the function will return entries that contain at least one
            of the `include` elements but none of the `exclude` elements.
        """
        if include is None:
            include = []
        if exclude is None:
            exclude = []

        if not include and not exclude:
            raise ValueError("At least one of `include` or `exclude` must be specified.")

        df = self.get_database("final")

        def contains_elements(elements_str, elements_list: list[str]) -> bool:
            """Checks if at least one element from elements_list is exactly present in Elements."""
            if isinstance(elements_str, str):  # Convert string to list if needed
                try:
                    elements = eval(elements_str)  # Safely parse string to list
                    if not isinstance(elements, list):
                        return False
                except:
                    return False
            else:
                elements = elements_str  # If it's already a list, use it as is

            return bool(set(elements) & set(elements_list))  # Exact match check

        # Apply filtering
        if include:
            df = df[df["Elements"].apply(lambda x: contains_elements(x, include))]
        if exclude:
            df = df[~df["Elements"].apply(lambda x: contains_elements(x, exclude))]

        return df'
    '''

    def filter_by_elements(
        self,
        include: list[str] = None,
        exclude: list[str] = None,
        stage: str = "final",
        save_filtered_db: bool = False,
    ) -> pd.DataFrame:
        """
        Filters the database entries based on the presence or absence of specified chemical elements or element groups.

        Args:
            include (list[str], optional): A list of chemical elements that must be present in the composition.
                                           - If None, no filtering is applied based on inclusion.
                                           - If an element group is passed using the format `"A-B"`, the material must
                                             contain *all* elements in that group.
            exclude (list[str], optional): A list of chemical elements that must not be present in the composition.
                                           - If None, no filtering is applied based on exclusion.
                                           - If an element group is passed using the format `"A-B"`, the material is
                                             removed *only* if it contains *all* elements in that group.
            stage (str, optional): The processing stage to retrieve the database from. Defaults to `"final"`.
                                   Possible values: `"raw"`, `"processed"`, `"final"`.
            save_filtered_db (bool, optional): If True, saves the filtered database back to `self.databases[stage]`
                                               and updates the stored database. Defaults to False.

        Returns:
            (pd.DataFrame): A filtered DataFrame containing only the entries that match the inclusion/exclusion
                            criteria.

        Raises:
            ValueError: If both `include` and `exclude` lists are empty.

        Notes:
            - If both `include` and `exclude` are provided, the function will return entries that contain at least one
              of the `include` elements but none of the `exclude` elements.
            - If an entry in `include` is in the format `"A-B"`, the material must contain all elements in that group.
            - If an entry in `exclude` is in the format `"A-B"`, the material is removed only if it contains *all*
              elements in that group.
            - If `save_filtered_db` is True, the filtered DataFrame is stored in `self.databases[stage]` and saved
              persistently.
        """

        if include is None:
            include = []
        if exclude is None:
            exclude = []

        if not include and not exclude:
            raise ValueError("At least one of `include` or `exclude` must be specified.")

        df = self.get_database(stage)

        def parse_elements(elements_str):
            """Convert the Elements column from string to an actual list."""
            if isinstance(elements_str, str):
                try:
                    elements = eval(elements_str)  # Convert string to list
                    if isinstance(elements, list):
                        return elements
                except Exception as e:
                    logger.warning(f"{e}")
            return elements_str if isinstance(elements_str, list) else []

        def contains_elements(elements, elements_list: list[str]) -> bool:
            """Check if an entry satisfies the element inclusion criteria."""
            material_elements = set(parse_elements(elements))  # Convert material elements to a set

            simple_elements = {e for e in elements_list if "-" not in e}  # Elements that can appear individually
            grouped_elements = [
                set(e.split("-")) for e in elements_list if "-" in e
            ]  # Element groups that must all be present

            # Check if any simple element is present
            simple_match = bool(material_elements & simple_elements) if simple_elements else False

            # Check if all elements in at least one group are present
            grouped_match = (
                any(group.issubset(material_elements) for group in grouped_elements) if grouped_elements else False
            )

            return simple_match or grouped_match  # Material passes if it satisfies either condition

        # Apply filtering
        if include:
            df = df[df["Elements"].apply(lambda x: contains_elements(x, include))]
        if exclude:
            df = df[~df["Elements"].apply(lambda x: contains_elements(x, exclude))]

        if save_filtered_db:
            self.databases[stage] = df
            self.save_database[stage]
            logger.info(f"Saved filtered database in stage {stage}.")

        return df

    def __repr__(self) -> str:
        """
        Text representation of the GNoMEDatabase instance.
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
        HTML representation of the GNoMEDatabase instance.
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
