from pathlib import Path

from loguru import logger
import pandas as pd
import torch
from tqdm import tqdm
import typer

from energy_gnome.config import (
    DATA_DIR,
    DEFAULT_E3NN_SETTINGS,
    DEFAULT_OPTIM_SETTINGS,
    DEFAULT_TRAINING_SETTINGS,
    MODELS_DIR,
)
from energy_gnome.dataset.base_dataset import BaseDatabase
from energy_gnome.models.e3nn.regressor import BaseRegressor, E3NNRegressor
from energy_gnome.utils.db_preprocessing.data_splitting import random_split


def eval_regressor(
    regressor_model: BaseRegressor,
    database: BaseDatabase,
    category: str,
    target_property: str,
    data_dir: Path = DATA_DIR,
    models_dir: Path = MODELS_DIR,
    logger=logger,
):
    logger.info("Evaluate regressors' performance")
    test_db = database.load_interim("testing")
    test_feat = regressor_model.db_featurizer(test_db)
    data = regressor_model.eval_regressor(test_feat)
    print(data[0])  # solo per zittire il pre-commit
