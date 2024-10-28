from loguru import logger
from mp_api.client import MPRester
import pandas as pd
from tqdm import tqdm

from energy_gnome.config import API_KEYS, CONFIG_YAML_FILE

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
def _get_mp_api_key() -> str:
    try:
        mp_api_key = API_KEYS["MP"]
    except KeyError as exc:
        logger.error("`MP` (Material Project) API KEY is missing in `config.yam` file")
        logger.error(f"Check {CONFIG_YAML_FILE}")
        raise exc
    return mp_api_key


def convert_my_query_to_dataframe(query: list, mute=False) -> pd.DataFrame:
    dict_query = []
    for d_ in tqdm(query, desc="converting InsertionElectrodeDoc documents", disable=mute):
        dict_query.append(d_.dict())
    database = pd.DataFrame(dict_query)
    return database


def get_material_by_id(ids: list, doc_prefix="", mute=False) -> list:
    mp_api_key = _get_mp_api_key()
    logger.debug(f"{doc_prefix} retriving {len(ids)} material from MP")
    with MPRester(mp_api_key, mute_progress_bars=mute) as mpr:
        docs = mpr.materials.summary.search(
            material_ids=ids,
            fields=MAT_FIELDS,
        )
    return docs
