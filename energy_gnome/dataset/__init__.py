from pathlib import Path  # noqa: F401

from energy_gnome.config import DATA_DIR
from energy_gnome.utils.logger_config import logger

from .cathodes import CathodeDatabase
from .raw_data import get_raw_cathode