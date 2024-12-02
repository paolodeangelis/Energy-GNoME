from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from energy_gnome.config import DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from energy_gnome.dataset import CathodeDatabase, PerovskiteDatabase

# app = typer.Typer()


def get_raw_cathode(
    data_dir: Path = DATA_DIR, working_ion="Li", battery_type="insertion", logger=logger
):
    cathode_db = CathodeDatabase(
        data_dir=data_dir, working_ion=working_ion, battery_type=battery_type
    )
    cathode_db.allow_raw_update()
    logger.info("[STEP 1] Retrieving models")
    db = cathode_db.retrieve_models(mute_progress_bars=False)
    cathode_db.compare_and_update(db, "raw")
    logger.info("[STEP 2] Retrieving and save charged material")
    materials_charge = cathode_db.retrieve_materials(
        stage="raw", charge_state="charge", mute_progress_bars=False
    )
    cathode_db.save_cif_files(
        stage="raw",
        materials_mp_query=materials_charge,
        charge_state="charge",
        mute_progress_bars=False,
    )
    logger.info("[STEP 3] Retrieving and saving discharge material")
    materials_discharge = cathode_db.retrieve_materials(
        stage="raw", charge_state="discharge", mute_progress_bars=False
    )
    cathode_db.save_cif_files(
        stage="raw",
        materials_mp_query=materials_discharge,
        charge_state="discharge",
        mute_progress_bars=False,
    )
    logger.info("[STEP 4] Saving database")
    cathode_db.save_database("raw")


def get_raw_perovskite(data_dir: Path = DATA_DIR, logger=logger):
    perovskite_db = PerovskiteDatabase(data_dir=data_dir)
    perovskite_db.allow_raw_update()
    logger.info("[STEP 1] Retrieving and saving materials")
    db, materials = perovskite_db.retrieve_materials(mute_progress_bars=False)
    perovskite_db.compare_and_update(db, "raw")
    logger.info("[STEP 2] Retrieving and saving CIF files")
    perovskite_db.save_cif_files(
        stage="raw",
        materials_mp_query=materials,
        mute_progress_bars=False,
    )
    logger.info("[STEP 3] Saving database")
    perovskite_db.save_database("raw")


# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     input_path: Path = RAW_DATA_DIR / "dataset.csv",
#     output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
#     # ----------------------------------------------
# ):
#     # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info("Processing dataset...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Processing dataset complete.")
#     # -----------------------------------------


# if __name__ == "__main__":
#     app()
