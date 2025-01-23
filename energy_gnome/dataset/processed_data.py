from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from energy_gnome.config import DATA_DIR
from energy_gnome.dataset import PerovskiteDatabase


def process_perovskite(data_dir: Path = DATA_DIR, logger=logger):
    perovskite_db = PerovskiteDatabase(data_dir=data_dir)
    logger.info("[STEP 1] Cleaning raw perovskite database")
    perovskite_db.process_database(band_gap_lower=0.0, band_gap_upper=2.5, clean_magnetic=True)
    perovskite_db.copy_cif_files("processed")
