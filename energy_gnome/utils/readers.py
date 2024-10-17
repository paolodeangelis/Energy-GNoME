import yaml

from energy_gnome.config import logger


@logger.catch
def read_yaml(filename: str) -> dict:
    with open(filename) as stream:
        try:
            logger.debug(f"Reading {filename}")
            content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return content
