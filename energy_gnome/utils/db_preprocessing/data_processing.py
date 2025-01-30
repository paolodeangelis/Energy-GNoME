from pathlib import Path

from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from energy_gnome.config import DATA_DIR
from energy_gnome.dataset import (
    BaseDatabase,
    MergedDatabase,
    MPDatabase,
    PerovskiteDatabase,
)


def process_perovskite(data_dir: Path = DATA_DIR, logger=logger):
    perovskite_db = PerovskiteDatabase(data_dir=data_dir)
    logger.info("[STEP 1] Cleaning raw perovskite database")
    perovskite_db.process_database(
        band_gap_lower=0.0, band_gap_upper=2.5, inplace=True, clean_magnetic=True
    )
    perovskite_db.copy_cif_files("processed")


def process_mp(category: str, mp_db: MPDatabase, logger=logger):
    if category == "perovskites":
        logger.info("[STEP 1] Removing cross-overlap between MP dabatase and perovskite database.")
        perovskite_db = PerovskiteDatabase()
        db = perovskite_db.load_database("raw")
        mp_database = mp_db.remove_cross_overlap("raw", db)
        logger.info(
            "[STEP 2] Cleaning raw MP database according to PerovskiteDatabase preproccessing rules."
        )
        mp_clean = perovskite_db.process_database(
            band_gap_lower=0.0,
            band_gap_upper=2.5,
            inplace=False,
            db=mp_database,
            clean_magnetic=True,
        )
        mp_db.databases["processed"] = mp_clean
        logger.info(f"Saving database.")
        mp_db.save_database("processed")


def merge_db(db_1: BaseDatabase, db_2: BaseDatabase, name: str, stage: str = "processed"):
    df_1 = db_1.get_database(stage)
    df_2 = db_2.get_database(stage)

    logger.info("[STEP 1] Concatenating the dataframes.")
    merged_df = pd.concat([df_1, df_2], ignore_index=True)

    logger.info("[STEP 2] Creating the MergedDatabase object.")
    merged_db = MergedDatabase(name=name)

    logger.info("[STEP 3] Saving the merged database.")
    merged_db.databases[stage] = merged_df
    merged_db.save_database(stage)
