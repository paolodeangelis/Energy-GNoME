from pathlib import Path  # noqa: F401

from dotenv import load_dotenv
from loguru import logger  # noqa: F401
from mp_api.client import MPRester  # noqa: F401
import pandas as pd

from energy_gnome.config import RAW_DATA_DIR

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
CATHODES_RAW_DATA_DIR = RAW_DATA_DIR / "cathodes"

WORKING_IONS = ["Li"]


BAT_FIELDS = [
    "battery_id",
    "thermo_type",
    "battery_formula",
    "working_ion",
    "num_steps",
    "max_voltage_step",
    "last_updated",
    "framework",
    "framework_formula",
    "elements",
    "nelements",
    "warnings",
    "formula_charge",
    "formula_discharge",
    "max_delta_volume",
    "average_voltage",
    "capacity_grav",
    "capacity_vol",
    "energy_grav",
    "energy_vol",
    "fracA_charge",
    "fracA_discharge",
    "stability_charge",
    "stability_discharge",
    "id_charge",
    "id_discharge",
    "adj_pairs",
    "material_ids",
]


def get_insertion_electrodes():
    pass


def get_batteries_model(working_ion: str, type="insertion") -> pd.DataFrame:
    pass
