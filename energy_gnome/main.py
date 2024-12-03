from pathlib import Path  # noqa: F401

from tqdm import tqdm  # noqa: F401
import typer
import os

from energy_gnome.config import (  # noqa: F401
    BATTERY_TYPES,
    DATA_DIR,
    LOG_MAX_WIDTH,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    WORKING_IONS,
)
from energy_gnome.dataset import get_raw_cathode, get_raw_perovskite
from energy_gnome.dataset.cathodes import CathodeDatabase
from energy_gnome.dataset.perovskites import PerovskiteDatabase
from energy_gnome.dataset.temp_convert import process_data_with_yaml
from energy_gnome.utils.logger_config import logger, setup_logging  # noqa: F401

# Create the Typer app
app = typer.Typer(help="Energy GNoME CLI.")


@app.command()
def datasets(
    energy_material: str = typer.Argument(..., help="Type of energy material (e.g., 'cathode')."),
    data_dir: Path = typer.Option(DATA_DIR, help="Directory path to save raw data."),
    logger_level: str = typer.Option("DEBUG", help="Set the logging level."),
) -> None:
    """
    Build datasets related to specified energy materials.

    This command retrieves raw dataset files related to battery materials, focusing on cathode datasets.
    It iterates over predefined battery types and working ions, fetching data according to the specified
    energy material type. Only cathode-related materials are currently supported.

    Args:
        energy_material (str): Specifies the type of energy material (e.g., "cathode").
        data_dir (Path): The directory path for saving raw data. Defaults to `DATA_DIR`.
        logger_level (str): Logging level for detailed debug information. Defaults to "DEBUG".
    """
    logger = setup_logging(level=logger_level)
    logger.info(" RAW DATASET ".center(LOG_MAX_WIDTH, "+"))

    if energy_material.lower() in ["cathode", "cathodes", "batteries", "battery"]:
        for battery_type in BATTERY_TYPES:
            for working_ion in WORKING_IONS:
                logger.info(
                    f"--- Getting cathodes for working ion '{working_ion}' and '{battery_type}' battery type ".ljust(
                        LOG_MAX_WIDTH, "-"
                    )
                )
                get_raw_cathode(
                    data_dir=data_dir,
                    logger=logger,
                    working_ion=working_ion,
                    battery_type=battery_type,
                )
    elif energy_material.lower() in ["photovoltaic", "photovoltaics", "perovskites", "perovskite"]:
        logger.info("--- Getting perovskites".ljust(LOG_MAX_WIDTH, "-"))
        get_raw_perovskite(
            data_dir=data_dir,
            logger=logger,
        )
    else:
        raise NotImplementedError(
            f"The database for the energy material '{energy_material}' is not supported."
        )


@app.command()
def pre_processing(something: str = typer.Argument(..., help="Data to preprocess.")) -> None:
    """
    Placeholder function for data preprocessing.

    This command will handle preprocessing tasks as required by the application in future versions.

    Args:
        something (str): Placeholder argument for data to be preprocessed.

    Returns:
        None
    """
    # TODO: Implement preprocessing logic
    logger.info("Preprocessing data...")
    pass

@app.command()
def convert(database_path: str = typer.Argument(help="Interim database to preprocess."), yaml_file: str = typer.Argument(help="Yaml configuration file.")):
    df, final_output_path = process_data_with_yaml(database_path, yaml_file) 
    out_dir = Path(os.path.dirname(final_output_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_json(final_output_path)


def main():
    app()


if __name__ == "__main__":
    main()
