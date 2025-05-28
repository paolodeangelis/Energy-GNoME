from pathlib import Path

from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from energy_gnome.config import DATA_DIR
from energy_gnome.dataset import MPDatabase, PerovskiteDatabase


def process_perovskite(data_dir: Path = DATA_DIR, logger=logger):
    perovskite_db = PerovskiteDatabase(data_dir=data_dir)
    logger.info("[STEP 1] Cleaning raw perovskite database")
    perovskite_db.process_database(band_gap_lower=0.0, band_gap_upper=2.5, inplace=True, clean_magnetic=True)
    perovskite_db.copy_cif_files("processed")


def process_mp(category: str, mp_db: MPDatabase, logger=logger):
    if category == "perovskites":
        logger.info("[STEP 1] Removing cross-overlap between MP dabatase and perovskite database.")
        perovskite_db = PerovskiteDatabase()
        perovskite_db.load_database("raw")
        db = perovskite_db.get_database("raw")
        mp_database = mp_db.remove_cross_overlap("raw", db)
        logger.info("[STEP 2] Cleaning raw MP database according to PerovskiteDatabase preproccessing rules.")
        mp_clean = perovskite_db.process_database(
            band_gap_lower=0.0,
            band_gap_upper=2.5,
            inplace=False,
            db=mp_database,
            clean_magnetic=True,
        )
        mp_db.databases["processed"] = mp_clean
        logger.info("Saving database.")
        mp_db.save_database("processed")
