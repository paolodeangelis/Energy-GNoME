import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
import torch

from energy_gnome.utils import read_yaml

# Load environment variables from .env file if it exists
load_dotenv()

# Article
DOI_ARTICLE = "doi:TBD"

# Paths
MODULE_ROOT = Path(__file__).resolve().parents[0]
PROJ_ROOT = Path(".").resolve()
# PROJ_ROOT = Path(__file__).resolve().parents[1]
# logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "raw"  # "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Supported working ions
WORKING_IONS = ["Li"]

# Supported battery types
BATTERY_TYPES = ["insertion"]

CONFIG_YAML_FILE = "config.yaml"

# API keys
if os.path.exists(CONFIG_YAML_FILE):
    API_KEYS = read_yaml(CONFIG_YAML_FILE)
else:
    logger.warning("`config.yaml` file missing, check README.md file")
    API_KEYS = {}

# Logger
LOG_MAX_WIDTH = 120

# Model settings
DEFAULT_E3NN_SETTINGS = {
    "n_committers": 4,
    "l_max": 2,  # maximum order of spherical harmonics (suggested: 2)
    "r_max": 5.0,  # cutoff radius for convolution (suggested: 5.0)
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
}

DEFAULT_E3NN_TRAINING_SETTINGS = {
    "n_epochs": 2,
    "batch_size": 4,
    "load_db": False,  # If the script will use a previously set-up database
}

DEFAULT_OPTIM_SETTINGS = {
    "lr": 0.005,  # Learning rate (suggested: 0.005)
    "wd": 0.05,  # Weight decay for AdamW optimizer (sort of L2 regularization) (suggested: 0.05)
}
