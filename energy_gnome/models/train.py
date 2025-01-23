from loguru import logger

from energy_gnome.dataset.base_dataset import BaseDatabase
from energy_gnome.models.e3nn.regressor import BaseRegressor


def fit_regressor(
    regressor_model: BaseRegressor,
    database: BaseDatabase,
    category: str,
    target_property: str,
    logger=logger,
):
    logger.info("[STEP 1] Regressor warmup")
    dataloader_dict = regressor_model.warmup(
        database,
        category,
        target_property,
    )

    logger.info("[STEP 2] Fit and save regressors")
    regressor_model.fit(dataloader_dict["train"], dataloader_dict["valid"])
