"""
Base Regressor Module for Energy Gnome Library.

This module defines the abstract base class `BaseRegressor`.
"""

from abc import ABC, abstractmethod
import json
from pathlib import Path
import time
from typing import Any

from ase.io import read
from loguru import logger
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric as tg
from tqdm import tqdm

from energy_gnome.config import (
    DEFAULT_E3NN_SETTINGS,
    DEFAULT_E3NN_TRAINING_SETTINGS,
    DEFAULT_OPTIM_SETTINGS,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)

# from energy_gnome.tools.model import PeriodicNetwork, train_regressor
from energy_gnome.models.regressor.e3nn_regressor_utils import (
    PeriodicNetwork,
    train_regressor,
)
from energy_gnome.tools.postprocessing import get_neighbors
from energy_gnome.tools.preprocessing import (
    build_data,
    get_encoding,
    train_valid_test_split,
)

DEFAULT_DTYPE = torch.float64
BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"
tqdm.pandas(bar_format=BAR_FORMAT)
torch.set_default_dtype(DEFAULT_DTYPE)


class BaseRegressor(ABC):
    def __init__(self, data_dir: Path = PROCESSED_DATA_DIR, models_dir: Path = MODELS_DIR):
        """
        Initialize the BaseRegressor with root data and models directories.

        Sets up the directory structure for accessing processed data
        and storing the trained models.

        Args:
            data_dir (Path, optional): Root directory path for reading data.
                                       Defaults to PROCESSED_DATA_DIR from config.
            models_dir (Path, optional): Root directory path for storing trained models.
                                         Defaults to MODELS_DIR from config.

        """
        self.data_dir = data_dir
        self.models_dir = models_dir
