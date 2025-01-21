import os
from pathlib import Path  # noqa: F401

from tqdm import tqdm  # noqa: F401
import typer

from energy_gnome.config import (  # noqa: F401
    BATTERY_TYPES,
    DATA_DIR,
    LOG_MAX_WIDTH,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    WORKING_IONS,
)
from energy_gnome.dataset import get_raw_cathode, get_raw_perovskite
from energy_gnome.dataset.cathodes import CathodeDatabase
from energy_gnome.dataset.perovskites import PerovskiteDatabase

# from energy_gnome.dataset.temp_convert import process_data_with_yaml
from energy_gnome.models.regressor import E3NNRegressor, fit_regressor
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
def pre_processing(
    energy_material: str = typer.Argument(..., help="Type of energy material (e.g., 'cathode')."),
    raw_data_dir: Path = typer.Option(DATA_DIR, help="Directory path to read raw data."),
    processed_data_dir: Path = typer.Option(
        DATA_DIR, help="Directory path to save processed data."
    ),
    logger_level: str = typer.Option("DEBUG", help="Set the logging level."),
) -> None:
    """
    Function for data preprocessing..

    Args:
        energy_material (str): Specifies the type of energy material (e.g., "cathode").
        raw_data_dir (Path): The directory path for reading raw data. Defaults to `DATA_DIR`.
        processed_data_dir (Path): The directory path for saving processed data. Defaults to `PROCESSED_DATA_DIR`.
        logger_level (str): Logging level for detailed debug information. Defaults to "DEBUG".

    Returns:
        None
    """
    logger = setup_logging(level=logger_level)
    logger.info(" RAW DATASET ".center(LOG_MAX_WIDTH, "+"))

    if energy_material.lower() in ["cathode", "cathodes", "batteries", "battery"]:
        for battery_type in BATTERY_TYPES:
            for working_ion in WORKING_IONS:
                logger.info(
                    f"--- Building regressors for working ion '{working_ion}' and '{battery_type}' battery type ".ljust(
                        LOG_MAX_WIDTH, "-"
                    )
                )
                # TODO: implement cathode preprocessing logic

    elif energy_material.lower() in ["photovoltaic", "photovoltaics", "perovskites", "perovskite"]:
        category = "perovskites"
        logger.info("--- Building regressors for perovskites".ljust(LOG_MAX_WIDTH, "-"))
        clean_perov_db(
            category=category,
            raw_data_dir=raw_data_dir,
            processed_data_dir=processed_data_dir,
            logger=logger,
        )
    else:
        raise NotImplementedError(
            f"The database for the energy material '{energy_material}' is not supported."
        )


@app.command()
def training(
    energy_material: str = typer.Argument(..., help="Type of energy material (e.g., 'cathode')."),
    model_type: str = typer.Argument(..., help="Type of model to train (e.g., 'regressor')."),
    data_dir: Path = typer.Option(
        PROCESSED_DATA_DIR, help="Directory path to access processed data."
    ),
    models_dir: Path = typer.Option(MODELS_DIR, help="Directory path to save trained models."),
    logger_level: str = typer.Option("DEBUG", help="Set the logging level."),
) -> None:
    """
    Trains ML models related to specified energy materials.

    Args:
        energy_material (str): Specifies the type of energy material (e.g., "cathode").
        model_type (str): Specifies the type of model to train (e.g., "regressor").
        data_dir (Path): The directory path for accessing processed data. Defaults to `PROCESSED_DATA_DIR`.
        models_dir (Path): The directory path for saving trained models. Defaults to `MODELS_DIR`.
        logger_level (str): Logging level for detailed debug information. Defaults to "DEBUG".
    """
    logger = setup_logging(level=logger_level)
    logger.info(" RAW DATASET ".center(LOG_MAX_WIDTH, "+"))

    if energy_material.lower() in ["cathode", "cathodes", "batteries", "battery"]:
        if model_type.lower() in ["regressor", "regressors", "regression"]:
            for battery_type in BATTERY_TYPES:
                for working_ion in WORKING_IONS:
                    logger.info(
                        f"--- Building regressors for working ion '{working_ion}' and '{battery_type}' battery type ".ljust(
                            LOG_MAX_WIDTH, "-"
                        )
                    )
        elif model_type.lower() in ["classifier", "classifiers", "classification"]:
            for battery_type in BATTERY_TYPES:
                for working_ion in WORKING_IONS:
                    logger.info(
                        f"--- Building classifiers for working ion '{working_ion}' and '{battery_type}' battery type ".ljust(
                            LOG_MAX_WIDTH, "-"
                        )
                    )
        else:
            raise NotImplementedError(f"'{model_type}' is not a valid model type entry.")
    elif energy_material.lower() in ["photovoltaic", "photovoltaics", "perovskites", "perovskite"]:
        category = "perovskites"
        target_property = "band_gap"
        if model_type.lower() in ["regressor", "regressors", "regression"]:
            logger.info("--- Building regressors for perovskites".ljust(LOG_MAX_WIDTH, "-"))
            fit_regressor(
                category=category,
                target_property=target_property,
                data_dir=data_dir,
                models_dir=models_dir,
                logger=logger,
            )
        elif model_type.lower() in ["classifier", "classifiers", "classification"]:
            logger.info("--- Building classifiers for perovskites".ljust(LOG_MAX_WIDTH, "-"))
            fit_classifier(
                category=category,
                data_dir=data_dir,
                models_dir=models_dir,
                logger=logger,
            )
    else:
        raise NotImplementedError(
            f"The database for the energy material '{energy_material}' is not supported."
        )


# @app.command()
# def convert(
#     database_path: str = typer.Argument(help="Interim database to preprocess."),
#     yaml_file: str = typer.Argument(help="Yaml configuration file."),
# ):
#     df, final_output_path = process_data_with_yaml(database_path, yaml_file)
#     out_dir = Path(os.path.dirname(final_output_path))
#     out_dir.mkdir(parents=True, exist_ok=True)
#     df.to_json(final_output_path)


def main():
    app()


if __name__ == "__main__":
    main()
