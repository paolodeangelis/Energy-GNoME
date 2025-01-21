from pathlib import Path

from loguru import logger
import torch
from tqdm import tqdm
import typer

from energy_gnome.config import (
    DEFAULT_E3NN_SETTINGS,
    DEFAULT_E3NN_TRAINING_SETTINGS,
    DEFAULT_OPTIM_SETTINGS,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)
from energy_gnome.models.regressor.e3nn_model import E3NNRegressor

# app = typer.Typer()

# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     features_path: Path = PROCESSED_DATA_DIR / "features.csv",
#     labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
#     model_path: Path = MODELS_DIR / "model.pkl",
#     # -----------------------------------------
# ):
#     # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info("Training some model...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Modeling training complete.")
#     # -----------------------------------------


# if __name__ == "__main__":
#     app()


def fit_regressor(
    category: str = "perovskites",
    target_property: str = "band_gap",
    data_dir: Path = PROCESSED_DATA_DIR,
    models_dir: Path = MODELS_DIR,
    logger=logger,
):
    regressor_model = E3NNRegressor(data_dir=data_dir, models_dir=models_dir)
    logger.info("[STEP 1] Loading settings")
    regressor_model.model_settings(category, target_property, DEFAULT_E3NN_SETTINGS)
    regressor_model.training_settings(DEFAULT_E3NN_TRAINING_SETTINGS)
    regressor_model.optimizer_settings(DEFAULT_OPTIM_SETTINGS)

    logger.info("[STEP 2] Load, build, and split datasets")
    database_nn = regressor_model.load_and_build_database()
    dataloader_dict, idx_dict, n_dict = regressor_model.random_split(
        database_nn, valid_size=0.2, test_size=0.05, seed=42
    )

    logger.info("[STEP 3] Configure the optimizer, lr_scheduler, and loss function for training")
    regressor_model.get_optimizer_scheduler_loss(
        optimizer_class=torch.optim.AdamW,
        scheduler_class=torch.optim.lr_scheduler.ExponentialLR,
        scheduler_settings={"gamma": 0.96},
        loss_function=torch.nn.L1Loss,
    )

    logger.info("[STEP 4] Build regressors")
    regressor_model.build_regressor(n_dict["train"])

    logger.info("[STEP 5] Train and save regressors")
    regressor_model.train_regressor(dataloader_dict["train"], dataloader_dict["valid"])

    logger.info("[STEP 6] Evaluate regressors' performance")
    data = regressor_model.eval_regressor(
        database_nn, idx_dict["train"], idx_dict["valid"], idx_dict["test"]
    )
    print(data[0])  # solo per zittire il pre-commit
