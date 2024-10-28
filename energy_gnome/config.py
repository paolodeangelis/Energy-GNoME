import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from energy_gnome.utils import read_yaml

# Load environment variables from .env file if it exists
load_dotenv()

# Article
DOI_ARTICLE = "doi:TBD"

# Paths
MODULE_ROOT = Path(__file__).resolve().parents[0]
PROJ_ROOT = Path(__file__).resolve().parents[1]
# logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Supported working ions
WORKING_IONS = ["Li"]

# Supported battery types
BATTERY_TYPES = ["insertion"]

CONFIG_YAML_FILE = MODULE_ROOT / "config.yaml"

# API keys
if os.path.exists(CONFIG_YAML_FILE):
    API_KEYS = read_yaml(CONFIG_YAML_FILE)
else:
    logger.warning("`config.yaml` file missing, check README.md file")
    API_KEYS = {}

# Logger
LOG_MAX_WIDTH = 120
