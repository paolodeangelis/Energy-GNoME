from loguru import logger

from energy_gnome.dataset.base_dataset import BaseDatabase
from energy_gnome.models.e3nn.regressor import BaseRegressor


def eval_regressor(
    regressor_model: BaseRegressor,
    database: BaseDatabase,
    subset: str,
    logger=logger,
):
    logger.info("[STEP 3] Evaluate regressors' performance")
    test_db = database.load_interim(subset)
    test_feat = regressor_model.db_featurizer(test_db)
    data = regressor_model.eval(test_feat)

    return data
