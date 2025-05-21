import json

from loguru import logger
import yaml


@logger.catch
def load_json(file_path):
    """Loads and returns data from a JSON file."""
    try:
        with open(file_path, encoding="utf-8") as file:
            return json.load(file)  # Parse JSON and return as dictionary/list
    except FileNotFoundError as exc:
        logger.error(f"Error: The file '{file_path}' was not found.")
        print(exc)
    except json.JSONDecodeError as exc:
        logger.error(f"Error: Failed to decode JSON from '{file_path}'.")
        print(exc)
    except Exception as exc:
        print(exc)


@logger.catch
def load_yaml(filename: str) -> dict:
    with open(filename) as stream:
        try:
            logger.debug(f"Reading {filename}")
            content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return content


@logger.catch
def save_yaml(data, file_path):
    """Saves data to a YAML file."""
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            yaml.dump(data, file, default_flow_style=False, allow_unicode=True)
        print(f"YAML file saved successfully at '{file_path}'")
    except Exception as e:
        print(f"Error: Failed to save YAML file.\nDetails: {e}")


@logger.catch
def to_unix(path):
    return path.replace("\\", "/") if "\\" in path else path
