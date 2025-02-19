"""
Base Regressor Module for Energy Gnome Library.

This module defines the abstract base class `BaseRegressor`.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import torch
import torch_geometric as tg
from tqdm import tqdm

from energy_gnome.config import FIGURES_DIR, MODELS_DIR
from energy_gnome.dataset import GNoMEDatabase

DEFAULT_DTYPE = torch.float64
BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"
tqdm.pandas(bar_format=BAR_FORMAT)
torch.set_default_dtype(DEFAULT_DTYPE)


class BaseRegressor(ABC):
    def __init__(
        self,
        model_name: str,
        target_property: str,
        models_dir: Path = MODELS_DIR,
        figures_dir: Path | str = FIGURES_DIR,
    ):
        """
        Initialize the BaseRegressor with root data and models directories.

        Sets up the directory structure for accessing processed data
        and storing the trained models.

        Args:
            model_name (str): ...
            data_dir (Path, optional): Root directory path for reading data.
                                       Defaults to PROCESSED_DATA_DIR from config.
            models_dir (Path, optional): Root directory path for storing trained models.
                                         Defaults to MODELS_DIR from config.

        """
        self.model_name = model_name
        self.target_property = target_property
        self.models_dir = Path(models_dir, model_name)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = Path(figures_dir, model_name)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.models: dict[str, torch.nn.Module] = {}
        self.device: str | None = None
        self.n_committers: int = 1
        self.batch_size: int = 1
        models_weights = self._find_model_states()
        if len(models_weights) > 0:
            n_model = len(models_weights)
            logger.warning(
                f"The folder {self.models_dir} already contains {n_model} trained models."
            )
            logger.warning(
                "Be careful, all changes (e.g. changing the inputs to methods as 'compile' and 'fit') will conflict with existing models!"
            )
            self.load_trained_models()

    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def _find_model_states(self):
        pass

    @abstractmethod
    def load_trained_models(self):
        pass

    @abstractmethod
    def create_dataloader(self):
        pass

    def _evaluate_unknown(
        self,
        dataloader: tg.loader.DataLoader,
    ) -> pd.DataFrame:

        prediction_nn = pd.DataFrame()

        for i in tqdm(range(self.n_committers), desc="models"):
            prediction_nn[f"regressor_{i}"] = np.empty((len(dataloader.dataset), 1)).tolist()

            self.models[f"model_{i}"].to(self.device)
            self.models[f"model_{i}"].eval()
            with torch.no_grad():
                i0 = 0
                for j, d in tqdm(enumerate(dataloader), total=len(dataloader)):
                    d.to(self.device)
                    output = self.models[f"model_{i}"](d)
                    # print([[k] for k in output.cpu().numpy()])
                    prediction_nn.loc[i0 : i0 + len(d.symbol) - 1, f"regressor_{i}"] = [
                        float(k) for k in output.cpu().numpy()
                    ]
                    i0 += len(d.symbol)

        prediction_nn["regressor_mean"] = prediction_nn[
            [f"regressor_{i}" for i in range(self.n_committers)]
        ].mean(axis=1)
        prediction_nn["regressor_std"] = prediction_nn[
            [f"regressor_{i}" for i in range(self.n_committers)]
        ].std(axis=1)

        return prediction_nn

    def predict(self, db: GNoMEDatabase, confidence_threshold: float = 0.5, save_final=True):
        logger.info(
            f"Discarding materials with classifier committee confidence threshold < {confidence_threshold}."
        )
        logger.info("Featurizing and loading database as `tg.loader.DataLoader`.")
        dataloader_db, _ = self.create_dataloader(db, confidence_threshold)
        logger.info("Predicting the target property for candidate specialized materials.")
        predictions = self._evaluate_unknown(dataloader_db)
        df = db.get_database("processed")
        screened = df[df["classifier_mean"] > confidence_threshold]
        predictions = pd.concat([screened.reset_index(), predictions], axis=1)

        if save_final:
            logger.info("Saving the final database.")
            db.databases["final"] = predictions.copy()
            db.save_database("final")

        return predictions


class BaseClassifier(ABC):
    def __init__(
        self, model_name: str, models_dir: Path = MODELS_DIR, figures_dir: Path | str = FIGURES_DIR
    ):
        """
        Initialize the BaseClassifier with root data and models directories.

        Sets up the directory structure for accessing processed data
        and storing the trained models.

        Args:
            model_name (str): ...
            data_dir (Path, optional): Root directory path for reading data.
                                       Defaults to PROCESSED_DATA_DIR from config.
            models_dir (Path, optional): Root directory path for storing trained models.
                                         Defaults to MODELS_DIR from config.

        """
        self.model_name = model_name
        self.target_property = "is_specialized"
        self.models_dir = Path(models_dir, model_name)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = Path(figures_dir, model_name)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.models: dict = {}
        self.n_committers: int = 10

    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def _find_model_states(self):
        pass

    @abstractmethod
    def load_trained_models(self):
        pass

    @abstractmethod
    def featurize_db(self):
        pass

    def _evaluate_unknown(self, df: pd.DataFrame) -> pd.DataFrame:
        predictions = pd.DataFrame(index=df.index)

        for i in tqdm(range(self.n_committers), desc="models"):
            predictions[f"classifier_{i}"] = self.models[f"model_{i}"].predict_proba(
                df.iloc[:, :-1]
            )[:, 1]
        predictions["classifier_mean"] = predictions[
            [f"classifier_{i}" for i in range(self.n_committers)]
        ].mean(axis=1)
        predictions["classifier_std"] = predictions[
            [f"classifier_{i}" for i in range(self.n_committers)]
        ].std(axis=1)
        return predictions

    def screen(self, db: GNoMEDatabase, save_processed=True):
        logger.info("Featurizing the database...")
        df_class = self.featurize_db(db)
        logger.info("Screening the database for specialized materials.")
        predictions = self._evaluate_unknown(df_class)
        gnome_df = db.get_database("raw")[:50]
        gnome_screened = pd.concat([gnome_df, predictions.reset_index(drop=True)], axis=1)
        gnome_screened.drop(columns=["is_specialized"], inplace=True)
        gnome_screened = gnome_screened[gnome_screened["classifier_mean"].notna()]

        if save_processed:
            logger.info("Saving the screened database.")
            db.databases["processed"] = gnome_screened.copy()
            db.save_database("processed")

        return gnome_screened
