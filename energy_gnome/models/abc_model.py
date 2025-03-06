"""
Base Regressor Module for Energy Gnome Library.

This module defines the abstract base class `BaseRegressor`.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from loguru import logger
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import torch
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
        Initialize the BaseRegressor with directories for storing models and figures.

        This class serves as a base for regression models, handling directory management
        for saving trained models and associated figures.

        Args:
            model_name (str): Name of the model, used to create subdirectories.
            target_property (str): The target property the model is trained to predict.
            models_dir (Path, optional): Directory for storing trained model weights.
                                        Defaults to MODELS_DIR from config.
            figures_dir (Path | str, optional): Directory for saving figures and visualizations.
                                                Defaults to FIGURES_DIR from config.

        Attributes:
            models_dir (Path): Path where model weights are stored.
            figures_dir (Path): Path where figures and visualizations are stored.
            models (dict[str, torch.nn.Module]): Dictionary of trained models.
            device (str | None): Computing device used for model training (e.g., "cpu" or "cuda").
            n_committers (int): Number of models in an ensemble (default is 1).
            batch_size (int): Batch size used for model training (default is 1).
        """

        self.model_name = model_name
        self.target_property = target_property

        self.models_dir = Path(models_dir, model_name) / "regressors"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.figures_dir = Path(figures_dir, model_name) / "regressors"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.models: dict[str, torch.nn.Module] = (
            {}
        )  # this will change with the addition of GBDT regressors
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
    def _find_model_states(self):
        pass

    @abstractmethod
    def load_trained_models(self):
        pass

    @abstractmethod
    def featurize_db(self):
        pass

    @abstractmethod
    def compile_(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def plot_parity(self):
        pass

    @abstractmethod
    def predict(self):
        pass


class BaseClassifier(ABC):
    def __init__(
        self, model_name: str, models_dir: Path = MODELS_DIR, figures_dir: Path | str = FIGURES_DIR
    ):
        """
        Initialize the BaseClassifier with directories for storing models and figures.

        This class serves as a base for classification models, handling directory management
        for saving trained models and associated figures.

        Args:
            model_name (str): Name of the model, used to create subdirectories.
            models_dir (Path, optional): Directory for storing trained model parameters.
                                        Defaults to `MODELS_DIR` from config.
            figures_dir (Path | str, optional): Directory for saving figures and visualizations.
                                                Defaults to `FIGURES_DIR` from config.

        Attributes:
            models_dir (Path): Path where model parameters are stored.
            figures_dir (Path): Path where figures and visualizations are stored.
            models (dict): Dictionary of trained models.
            n_committers (int): Number of models in an ensemble (default is 1).
        """
        self.model_name = model_name
        self.target_property = "is_specialized"

        self.models_dir = Path(models_dir, model_name) / "classifiers"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.figures_dir = Path(figures_dir, model_name) / "classifiers"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.models: dict[str, GradientBoostingClassifier] = {}
        self.n_committers: int = 1

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
    def _find_model_states(self):
        pass

    @abstractmethod
    def load_trained_models(self):
        pass

    @abstractmethod
    def featurize_db(self):
        pass

    @abstractmethod
    def compile_(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def plot_performance(self):
        pass

    @abstractmethod
    def screen(self):
        pass
