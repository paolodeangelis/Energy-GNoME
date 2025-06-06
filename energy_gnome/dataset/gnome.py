import gc
import multiprocessing
import os
from pathlib import Path
import shutil as sh
import subprocess

import pandas as pd
from tqdm.auto import tqdm

from energy_gnome.config import DATA_DIR, EXTERNAL_DATA_DIR  # noqa:401
from energy_gnome.dataset.base_dataset import BaseDatabase
from energy_gnome.exception import ImmutableRawDataError
from energy_gnome.utils.logger_config import logger


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
        # self.database_paths["raw"] = self.data_dir / "raw" / "gnome" / "database.json"

        self._gnome = pd.DataFrame()
        self._is_specialized = False
        self.external_gnome_dir = self.data_dir / Path(EXTERNAL_DATA_DIR) / "gdm_materials_discovery" / "gnome_data"
        self.load_all()

    def retrieve_materials(self) -> pd.DataFrame:
        """
        TBD (after implementing the fetch routine)
        """
        csv_db = pd.read_csv(self.external_gnome_dir / "stable_materials_summary.csv", index_col=0)
        csv_db = csv_db.rename(columns={"MaterialId": "material_id", "Reduced Formula": "formula_pretty"})

        return csv_db

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

                self.databases[stage] = new_db.copy()
                self.save_database(stage)
        else:
            logger.info("No new items found. No update required.")

    def save_cif_files(self) -> None:
        """
        Extract and store CIF structure files, and update the database with their paths.

        This method unzips a compressed archive of CIF files (`by_id.zip`) into a structured
        directory (`raw/structures/by_id`) using OS-native unzipping tools in an isolated
        subprocess for robustness and performance. It then updates the `cif_path` column of
        the raw database to reflect the location of the extracted files.

        Notes:
            - On Windows, this method requires **7-Zip** to be installed and available via the command line.
            - The unzipping process is done in a separate process to ensure isolation and avoid crashes
            from subprocess errors or memory issues.
            - If the output folder already exists, it will be **fully deleted** and recreated.
            - To avoid slowdowns in environments that continuously index files (e.g., VSCode),
            it is recommended to **exclude the `by_id/` subfolder** from indexing by adding the
            following to `.vscode/settings.json`:

            ```json
            {
            "files.watcherExclude": {
                "**/by_id/**": true
            },
            "search.exclude": {
                "**/by_id/**": true
            },
            "files.exclude": {
                "**/by_id/**": true
            }
            }
            ```

        Raises:
            RuntimeError: If the unzip process fails in the isolated subprocess.
            Exception: If any other error occurs during unzipping or path updating.

        Logs:
            - WARN: If the output folder already exists and is being deleted.
            - INFO: About the unzip source and destination.
            - ERROR: If an error occurs during the unzip process.
            - SUCCESS: When CIF extraction and database updating complete successfully.
        """
        zip_path = self.external_gnome_dir / "by_id.zip"
        output_path = self.database_directories["raw"] / "structures"

        # Clear and recreate directory
        if output_path.exists():
            logger.warning(f"Cleaning {output_path}")
            logger.warning("This may take a while...")
            sh.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=False)

        extracted_dir = output_path / "by_id"

        logger.info("Unzipping structures from")
        logger.info(f"{zip_path}")
        logger.info("to")
        logger.info(f"{extracted_dir}")

        try:
            self._unzip_with_isolation(zip_path, output_path)
            logger.info("Extraction complete!")
        except Exception as e:
            logger.error(f"Error during unzip: {e}")
            return

        # Update Database
        df = self.get_database("raw")
        df = self._update_cif_paths_with_tqdm(df, extracted_dir)

        self.save_database("raw")
        del df
        gc.collect()
        logger.info("CIF files saved and database updated successfully.")

    def _unzip_native(self, zip_path, output_path):
        if os.name == "nt":
            cmd = ["7z", "x", str(zip_path), f"-o{output_path}", "-y"]
        else:
            cmd = ["unzip", "-o", str(zip_path), "-d", str(output_path)]

        subprocess.run(cmd, check=True)

    def _unzip_with_isolation(self, zip_path, output_path):
        p = multiprocessing.Process(target=self._unzip_native, args=(zip_path, output_path))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Unzipping failed with exit code {p.exitcode}")

    def _update_cif_paths_with_tqdm(self, df: pd.DataFrame, output_path: Path, chunk_size: int = 10000) -> pd.DataFrame:
        """
        Updates the 'cif_path' column in chunks with a tqdm progress bar.
        """
        total_rows = len(df)

        # Calculate total chunks
        with tqdm(total=total_rows, desc="Updating CIF paths", unit="file") as pbar:
            for i in range(0, total_rows, chunk_size):
                end = min(i + chunk_size, total_rows)
                df.loc[i : end - 1, "cif_path"] = (
                    output_path.as_posix() + "/" + df.loc[i : end - 1, "material_id"].astype(str) + ".CIF"
                )
                pbar.update(end - i)

        return df

    def remove_cross_overlap(
        self,
        stage: str,
        df: pd.DataFrame,
        inplace: bool = False,
        new_name: str = "gnome_clean",
    ) -> pd.DataFrame:
        """
        Remove entries in the unexplored database that overlap with a category-specific database.

        This function identifies and removes material entries that exist in both the unexplored
        and category-specific databases for a given processing stage.

        Args:
            stage (str): Processing stage (`raw`, `processed`, `final`).
            df (pd.DataFrame): The category-specific database to compare with the unexplored database.
            save_db (bool): Whether to save the filtered database back into the internal store.

        Returns:
            pd.DataFrame: The filtered unexplored database with overlapping entries removed.

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
        gnome_df = self.get_database(stage)

        # Use set for fast lookup
        overlap_set = set(df["formula_pretty"])

        # Use vectorized filtering
        mask = ~gnome_df["formula_pretty"].isin(overlap_set)
        to_drop = (~mask).sum()
        logger.info(f"{to_drop} overlapping items to drop.")

        gnome_df_no_overlap = gnome_df.loc[mask].reset_index(drop=True)

        if inplace:
            self.databases[stage] = gnome_df_no_overlap.copy()
            self.save_database(stage)
        else:
            return gnome_df_no_overlap

    def filter_by_elements(
        self, include: list[str] = None, exclude: list[str] = None, stage: str = "final", save_filtered_db: bool = False
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
