"""
Material Project API Utilities Module for Energy Gnome Library.
"""

import os

from mp_api.client import MPRester
from mp_api.client.core import MPRestError
import pandas as pd
from tqdm import tqdm

from energy_gnome.config import API_KEYS, CONFIG_YAML_FILE

from .logger_config import logger

MAT_FIELDS = [
    "formula_pretty",
    "volume",
    "density",
    "material_id",
    "structure",
    "energy_per_atom",
    "formation_energy_per_atom",
    "is_stable",
]


@logger.catch
def get_mp_api_key() -> str:
    try:
        mp_api_key = API_KEYS["MP"]
        logger.debug("MP API key retrieved successfully.")
    except KeyError:
        mp_api_key = os.getenv("MP_API_KEY")
        if not mp_api_key:
            logger.error("`MP` (Material Project) API KEY is missing in `config.yam` file")
            logger.error(f"Check {CONFIG_YAML_FILE}")
            logger.error("`MP` (Material Project) API KEY environment variable MP_API_KEY is not set")
            logger.error("At least one option should be used")
            raise OSError("MP_API_KEY environment variable is not set.")
        logger.debug("MP API key retrieved successfully.")
    return mp_api_key


def convert_my_query_to_dataframe(query: list, mute_progress_bars=False) -> pd.DataFrame:
    dict_query = []
    for d_ in tqdm(query, desc="converting material documents", disable=mute_progress_bars):
        dict_query.append(d_.dict())
    dataframe = pd.DataFrame(dict_query)
    return dataframe


def get_material_by_id(ids: list, doc_prefix="", mute_progress_bars=False) -> list:
    mp_api_key = get_mp_api_key()
    logger.debug(f"{doc_prefix} retriving {len(ids)} material from MP")
    with MPRester(mp_api_key, mute_progress_bars=mute_progress_bars) as mpr:
        try:
            docs = mpr.materials.summary.search(
                material_ids=ids,
                fields=MAT_FIELDS,
            )
            logger.success(f"MP query successful with {len(docs)} results.")
        except MPRestError as e:
            logger.error(f"Material Project API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during MP query: {e}")
            raise
    return docs
