# energy_gnome/data/base_dataset.py

import os

try:
    import win32file
except ImportError:
    pass

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
import shutil as sh
from typing import Any

from loguru import logger
from numpy.random import PCG64, Generator
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from energy_gnome.config import DATA_DIR
from energy_gnome.exception import DatabaseError, ImmutableRawDataError, MissingData

from .utils import make_link, random_split


class BaseDatabase(ABC):
    """
    Abstract base class for managing a structured database system with multiple
    processing stages and data subsets.

    This class provides a standardized framework for handling data across different
    stages of processing (`raw`, `processed`, `final`). It ensures proper directory
    structure, initializes database placeholders, and offers an interface for
    subclassing specialized database implementations.

    Attributes:
        name (str): The name of the database instance.
        data_dir (Path): Root directory where database files are stored.
        processing_stages (list[str]): The main stages of data processing.
        interim_sets (list[str]): Subsets within the training pipelines (e.g., training, validation).
        database_directories (dict[str, Path]): Mapping of processing stages to their respective directories.
        database_paths (dict[str, Path]): Paths to database files for each processing stage.
        databases (dict[str, pd.DataFrame]): Data storage for each processing stage.
        subset (dict[str, pd.DataFrame]): Storage for subsets like training, validation, and testing.
        _update_raw (bool): Flag indicating whether raw data should be updated.
        _is_specialized (bool): Indicates whether a subclass contains materials for specialized energy applications.
    """

    def __init__(self, name: str, data_dir: Path = DATA_DIR):
        """
        Initializes the BaseDatabase instance.

        Sets up the directory structure for storing data across different processing
        stages (`raw`, `processed`, `final`) and initializes empty Pandas DataFrames
        for managing the data.

        Args:
            name (str): The name of the database instance.
            data_dir (Path, optional): Root directory path for storing database files.
                Defaults to `DATA_DIR`.
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
        self.database_directories = {stage: self.data_dir / stage / self.name for stage in self.processing_stages}
        for stage_dir in self.database_directories.values():
            stage_dir.mkdir(parents=True, exist_ok=True)

        self.database_paths = {
            stage: dir_path / "database.json" for stage, dir_path in self.database_directories.items()
        }

        self.databases = {stage: pd.DataFrame() for stage in self.processing_stages}
        self.subset = {subset: pd.DataFrame() for subset in self.interim_sets}
        self._update_raw = False
        self._is_specialized = False

    def allow_raw_update(self):
        """
        Enables modifications to the `raw` data stage.

        This method sets the internal flag `_update_raw` to `True`, allowing changes
        to be made to the raw data without raising an exception.

        Warning:
            Use with caution, as modifying raw data can impact data integrity.
        """
        self._update_raw = True

    def compare_databases(self, new_db: pd.DataFrame, stage: str) -> pd.DataFrame:
        """
        Compare two databases and identify new entry IDs.

        This method compares an existing database (loaded from the specified stage) with a new database.
        It returns the entries from the new database that are not present in the existing one.

        Args:
            new_db (pd.DataFrame): New database to compare.
            stage (str): Processing stage ("raw", "processed", "final").

        Returns:
            pd.DataFrame: Subset of `new_db` containing only new entry IDs.

        Logs:
            - DEBUG: The number of new entries found.
            - WARNING: If the old database is empty and nothing can be compared.
        """
        old_db = self.get_database(stage=stage)
        if not old_db.empty:
            new_ids_set = set(new_db["material_id"])
            old_ids_set = set(old_db["material_id"])
            new_ids_only = new_ids_set - old_ids_set
            logger.debug(f"Found {len(new_ids_only)} new IDs in the new database.")
            return new_db[new_db["material_id"].isin(new_ids_only)]
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

        This method saves a backup of the existing database before updating it with new data.
        It also logs the changes detected by comparing the old and new databases, storing the
        details in a changelog file.

        Args:
            old_db (pd.DataFrame): The existing database before updating.
            new_db (pd.DataFrame): The new database with updated entries.
            differences (pd.Series): A series containing the material IDs of entries that differ.
            stage (str): The processing stage ("raw", "processed", "final") for which the backup
                and changelog are being maintained.

        Raises:
            ValueError: If an invalid `stage` is provided.
            OSError: If there is an issue writing to the backup or changelog files.

        Logs:
            - ERROR: If an invalid stage is provided.
            - DEBUG: When the old database is successfully backed up.
            - ERROR: If the backup process fails.
            - DEBUG: When the changelog is successfully updated with differences.
            - ERROR: If updating the changelog fails.
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
            f"{'ID':<15}{'Formula':<30}{'Last Updated':<25}\n" + "-" * 70 + "\n"
        )

        # Set index for faster lookup
        new_db_indexed = new_db.set_index("material_id")

        # Process differences efficiently
        if "last_updated" in new_db_indexed.columns:
            changes = [
                f"{identifier:<15}{new_db_indexed.at[identifier, 'formula_pretty'] if identifier in new_db_indexed.index else 'N/A':<30}"
                f"{new_db_indexed.at[identifier, 'last_updated'] if identifier in new_db_indexed.index else 'N/A':<25}\n"
                for identifier in differences["material_id"]
            ]
        else:
            changes = [
                f"{identifier:<15}{new_db_indexed.at[identifier, 'formula_pretty'] if identifier in new_db_indexed.index else 'N/A':<30}"
                for identifier in differences["material_id"]
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
                self.databases[stage] = new_db
                self.save_database(stage)
        else:
            logger.info("No new items found. No update required.")

    @abstractmethod
    def retrieve_materials(self) -> list[Any]:
        pass

    def save_cif_files(self) -> None:
        pass

    def copy_cif_files(self, stage: str, mute_progress_bars: bool = True) -> None:
        """
        Copy CIF files from the raw stage to another processing stage.

        This method transfers CIF files from the `raw` stage directory to the specified
        processing stage (`processed` or `final`). It ensures that existing CIF files in
        the target directory are cleared before copying and updates the database with
        the new file paths.

        Args:
            stage (str): The processing stage to copy CIF files to (`processed`, `final`).
            mute_progress_bars (bool, optional): If True, disables progress bars. Defaults to True.

        Raises:
            ValueError: If the stage argument is `raw`, as copying is only allowed from `raw` to other stages.
            MissingData: If the raw CIF directory does not exist or is empty.

        Logs:
            - WARNING: If the target directory is cleaned or CIF files are missing for some materials.
            - ERROR: If a CIF file fails to copy.
            - INFO: When CIF files are successfully copied and the database is updated.
        """
        if stage == "raw":
            logger.error("Stage argument cannot be 'raw'. You can only copy from 'raw' to other stages.")
            raise ValueError("Stage argument cannot be 'raw'.")

        source_dir = self.database_directories["raw"] / "structures"
        saving_dir = self.database_directories[stage] / "structures"

        # Clean the target directory if it exists
        if saving_dir.exists():
            logger.warning(f"Cleaning the content in {saving_dir}")
            sh.rmtree(saving_dir)

        # Check if source directory exists and is not empty
        cif_files = {file.stem for file in source_dir.glob("*.cif")}  # Set of existing CIF filenames
        if not cif_files:
            logger.warning(f"The raw CIF directory does not exist or is empty. Check: {source_dir}")
            raise MissingData(f"The raw CIF directory does not exist or is empty. Check: {source_dir}")

        # Create the target directory
        saving_dir.mkdir(parents=True, exist_ok=False)

        # Create an index mapping for fast row updates
        db_stage = self.databases[stage].set_index("material_id")
        db_stage["cif_path"] = pd.NA  # Initialize empty column

        missing_ids = []
        for material_id in tqdm(
            self.databases[stage]["material_id"],
            desc=f"Copying materials ('raw' -> '{stage}')",
            disable=mute_progress_bars,
        ):
            if material_id not in cif_files:
                missing_ids.append(material_id)
                continue  # Skip missing files

            source_cif_path = source_dir / f"{material_id}.cif"
            cif_path = saving_dir / f"{material_id}.cif"

            try:
                sh.copy2(source_cif_path, cif_path)
                db_stage.at[material_id, "cif_path"] = str(cif_path)  # Direct assignment

            except Exception as e:
                logger.error(f"Failed to copy CIF for Material ID {material_id}: {e}")
                continue  # Skip to next material instead of stopping execution

        # Restore the updated database index
        self.databases[stage] = db_stage.reset_index()

        # Log missing files once
        if missing_ids:
            logger.warning(f"Missing CIF files for {len(missing_ids)} material IDs.")

        # Save the updated database
        self.save_database(stage)
        logger.info(f"CIF files copied to stage '{stage}' and database updated successfully.")

    def get_database(self, stage: str, subset: str | None = None) -> pd.DataFrame:
        """
        Retrieves the database for a specified processing stage or subset.

        This method returns the database associated with the given `stage`. If the stage
        is `raw`, `processed`, or `final`, it retrieves the corresponding database.
        If `stage` is `interim`, a specific `subset` (e.g., `training`, `validation`, `testing`)
        must be provided.

        Args:
            stage (str): The processing stage to retrieve. Must be one of
                `raw`, `processed`, `final`, or `interim`.
            subset (Optional[str]): The subset to retrieve when `stage` is `interim`.
                Must be one of `training`, `validation`, or `testing`.

        Returns:
            pd.DataFrame: The requested database or subset.

        Raises:
            ValueError: If an invalid `stage` is provided.
            ValueError: If `stage` is `interim` but an invalid `subset` is specified.

        Logs:
            - ERROR: If an invalid `stage` is provided.
            - WARNING: If the retrieved database is empty.
        """
        if stage not in self.processing_stages + ["interim"]:
            logger.error(f"Invalid stage: {stage}. Must be one of {self.processing_stages + ['interim']}.")
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

    def load_database(self, stage: str) -> None:
        """
        Loads the existing database for a specified processing stage.

        This method retrieves the database stored in a JSON file for the given
        processing stage (`raw`, `processed`, or `final`). If the file exists,
        it loads the data into a pandas DataFrame. If the file is missing,
        a warning is logged, and an empty DataFrame remains in place.

        Args:
            stage (str): The processing stage to load. Must be one of
                `raw`, `processed`, or `final`.

        Raises:
            ValueError: If `stage` is not one of the predefined processing stages.

        Logs:
            ERROR: If an invalid `stage` is provided.
            DEBUG: If a database is successfully loaded.
            WARNING: If the database file is not found.
        """
        if stage not in self.processing_stages:
            logger.error(f"Invalid stage: {stage}. Must be one of {self.processing_stages}.")
            raise ValueError(f"stage must be one of {self.processing_stages}.")

        db_path = self.database_paths[stage]
        if db_path.exists():
            self.databases[stage] = pd.read_json(db_path)
            if stage == "raw":
                self.databases[stage]["is_specialized"] = self._is_specialized
            logger.debug(f"Loaded existing database from {db_path}.")
        else:
            logger.warning(f"Not found at {db_path}.")

    def _load_interim(self, subset: str = "training", model_type: str = "regressor") -> pd.DataFrame:
        """
        Load the existing interim databases.

        This method attempts to load an interim database corresponding to the specified
        subset and model type. If the database file is found, it is loaded into a pandas
        DataFrame. If not found, a warning is logged, and an empty DataFrame is returned.

        Args:
            subset (str): The subset of the interim dataset to load (`training`, `validation`, `testing`).
            model_type (str, optional): The type of model associated with the data (`regressor`, `classifier`).
                Defaults to "regressor".

        Returns:
            (pd.DataFrame): The loaded database if found, otherwise an empty DataFrame.

        Raises:
            ValueError: If the provided `subset` is not one of the allowed interim sets.

        Logs:
            - ERROR: If an invalid subset is provided.
            - DEBUG: If an existing database is successfully loaded.
            - WARNING: If no database file is found.
        """

        if subset not in self.interim_sets:
            logger.error(f"Invalid set: {subset}. Must be one of {self.interim_sets}.")
            raise ValueError(f"set must be one of {self.interim_sets}.")

        db_name = subset + "_db.json"
        db_path = self.data_dir / "interim" / self.name / model_type / db_name
        if db_path.exists():
            self.subset[subset] = pd.read_json(db_path)
            logger.debug(f"Loaded existing database from {db_path}")
        else:
            logger.warning(f"No existing database found at {db_path}")
        return self.subset[subset]

    def load_regressor_data(self, subset: str = "training"):
        """
        Load the interim dataset for a regression model.

        This method retrieves the specified subset of the interim dataset specifically for
        regression models by internally calling `_load_interim`.

        Args:
            subset (str, optional): The subset of the dataset to load (`training`, `validation`, `testing`).
                Defaults to "training".

        Returns:
            (pd.DataFrame): The loaded regression dataset or an empty DataFrame if not found.

        Raises:
            ValueError: If the provided `subset` is not one of the allowed interim sets.

        Logs:
            - ERROR: If an invalid subset is provided.
            - DEBUG: If an existing database is successfully loaded.
            - WARNING: If no database file is found.
        """
        return self._load_interim(subset=subset, model_type="regressor")

    def load_classifier_data(self, subset: str = "training"):
        """
        Load the interim dataset for a classification model.

        This method retrieves the specified subset of the interim dataset specifically for
        classification models by internally calling `_load_interim`.

        Args:
            subset (str, optional): The subset of the dataset to load (`training`, `testing`).
                Defaults to "training".

        Returns:
            (pd.DataFrame): The loaded regression dataset or an empty DataFrame if not found.

        Raises:
            ValueError: If the provided `subset` is not one of the allowed interim sets.

        Logs:
            - ERROR: If an invalid subset is provided.
            - DEBUG: If an existing database is successfully loaded.
            - WARNING: If no database file is found.
        """
        return self._load_interim(subset=subset, model_type="classifier")

    def load_all(self):
        """
        Loads the databases for all processing stages and subsets.

        This method sequentially loads the databases for all predefined processing
        stages (`raw`, `processed`, `final`). Additionally, it loads both regressor
        and classifier data for all interim subsets (`training`, `validation`, `testing`).

        Calls:
            - `load_database(stage)`: Loads the database for each processing stage.
            - `load_regressor_data(subset)`: Loads regressor-specific data for each subset.
            - `load_classifier_data(subset)`: Loads classifier-specific data for each subset.
        """
        for stage in self.processing_stages:
            self.load_database(stage)
        for subset in self.interim_sets:
            self.load_regressor_data(subset)
            self.load_classifier_data(subset)

    def save_database(self, stage: str) -> None:
        """
        Saves the current state of the database to a JSON file.

        This method serializes the database DataFrame for the specified processing
        stage (`raw`, `processed`, or `final`) and writes it to a JSON file. If an
        existing file is present, it is removed before saving the new version.

        Args:
            stage (str): The processing stage to save. Must be one of
                `raw`, `processed`, or `final`.

        Raises:
            ValueError: If `stage` is not one of the predefined processing stages.
            OSError: If an error occurs while writing the DataFrame to the file.

        Logs:
            - ERROR: If an invalid `stage` is provided.
            - INFO: If the database is successfully saved.
            - ERROR: If the save operation fails.
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

    def random_downsample(
        self,
        size: int,
        new_name: str,
        stage: str,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Build a reduced database by sampling random entries from an existing database.

        This method creates a new database by randomly sampling a specified number of entries
        from the given stage (`raw`, `processed`, or `final`) of the existing database. The
        new database is saved and returned as a new instance.

        Args:
            size (int): The number of entries for the reduced database.
            new_name (str): The name for the new reduced database.
            stage (str): The processing stage (`raw`, `processed`, or `final`) to sample from.
            seed (int, optional): The random seed used to generate reproducible samples. Defaults to 42.

        Returns:
            (pd.DataFrame): The reduced database as a pandas DataFrame with the sampled entries.

        Raises:
            ValueError: If `size` is 0, indicating that an empty database is being created.

        Logs:
            ERROR: If the database size is set to 0.
            INFO: If the new reduced database is successfully created and saved.
        """
        new_database = self.__class__(name=new_name, data_dir=self.data_dir)

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
                subset_path_son = new_database.data_dir / "interim" / new_database.name / (subset + "_db.json")
                if not subset_path_son.parent.exists():
                    subset_path_son.parent.mkdir(parents=True, exist_ok=True)
                make_link(subset_path_father, subset_path_son)

        df = new_database.databases[stage].copy()
        n_all = len(df)
        rng = Generator(PCG64(seed))
        row_i_all = rng.choice(n_all, size, replace=False)
        new_database.databases[stage] = df.iloc[row_i_all, :].reset_index(drop=True)

        new_database.save_database(stage)

        return new_database.databases[stage]

    def save_split_db(self, database_dict: dict, model_type: str = "regressor") -> None:
        """
        Saves the split databases (training, validation, testing) into JSON files.

        This method saves the split databases into individual files in the designated
        directory for the given model type (e.g., `regressor`, `classifier`). It checks
        whether the databases are empty before saving. If any of the databases are empty,
        it logs a warning or raises an error depending on the subset (training, validation, testing).

        Args:
            database_dict (dict): A dictionary containing the split databases (`train`,
                `valid`, `test`) as pandas DataFrames.
            model_type (str, optional): The model type for which the splits are being saved
                (e.g., `"regressor"`, `"classifier"`). Defaults to `"regressor"`.

        Returns:
            None

        Raises:
            DatabaseError: If the training dataset is empty
                when attempting to save.

        Logs:
            INFO: When a database is successfully saved to its designated path.
            WARNING: When the validation or testing database is empty.
            ERROR: If any dataset is empty and it is the `train` subset.
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
    ) -> None:
        """
        Splits the processed database into training, validation, and test sets for regression tasks.

        This method divides the database into three subsets: training, validation, and test. It
        either performs a random split with or without balancing the frequency of chemical
        species across the splits. If `balance_composition` is True, it ensures that
        elements appear in approximately equal proportions in each subset. The split sizes for
        validation and test sets can be customized.

        Args:
            target_property (str): The property used for the regression task (e.g., a material
                property like "energy").
            valid_size (float, optional): The proportion of the data to use for the validation set.
                Defaults to 0.2.
            test_size (float, optional): The proportion of the data to use for the test set.
                Defaults to 0.05.
            seed (int, optional): The random seed for reproducibility. Defaults to 42.
            balance_composition (bool, optional): Whether to balance the frequency of chemical species
                across the subsets. Defaults to True.
            save_split (bool, optional): Whether to save the resulting splits as files. Defaults to False.

        Returns:
            None

        Raises:
            ValueError: If the sum of `valid_size` and `test_size` exceeds 1.

        Logs:
            INFO: If the dataset is successfully split.
            ERROR: If the sum of `valid_size` and `test_size` is greater than 1.
        """
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
                train_, test_ = train_test_split(self.get_database("processed"), test_size=test_size, random_state=seed)
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
        balance_composition: bool = True,
        save_split: bool = False,
    ) -> None:
        """
        Splits the processed database into training and test sets for classification tasks.

        This method divides the database into two subsets: training and test. It always stratifies
        the split based on the target property (`is_specialized`). If `balance_composition` is True,
        it additionally balances the frequency of chemical species across the training and test sets.
        The size of the test set can be customized with the `test_size` argument.

        Args:
            test_size (float, optional): The proportion of the data to use for the test set.
                Defaults to 0.2.
            seed (int, optional): The random seed for reproducibility. Defaults to 42.
            balance_composition (bool, optional): Whether to balance the frequency of chemical
                species across the training and test sets in addition to stratifying by the
                target property (`is_specialized`). Defaults to True.
            save_split (bool, optional): Whether to save the resulting splits as files. Defaults to False.

        Returns:
            None

        Raises:
            ValueError: If the database is empty or invalid.

        Logs:
            INFO: If the dataset is successfully split.
            ERROR: If the dataset is invalid or empty.
        """
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
            train_, test_ = train_test_split(self.get_database("processed"), test_size=test_size, random_state=seed)
            db_dict = {"train": train_, "test": test_}

        if save_split:
            self.save_split_db(db_dict, "classifier")

        self.subset = db_dict

    def __repr__(self) -> str:
        """
        Text representation of the BaseDatabase instance.
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
        HTML representation of the BaseDatabase instance.
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
            "</div>"
        )
        return html
