from pathlib import Path  # noqa: F401

from loguru import logger  # noqa: F401
from tqdm import tqdm  # noqa: F401
import typer

from energy_gnome.config import PROCESSED_DATA_DIR, RAW_DATA_DIR  # noqa: F401
from energy_gnome.dataset import main

app = typer.Typer(help="Awesome CLI user manager.")


@app.command()
def datasets():
    main


if __name__ == "__main__":
    app()
