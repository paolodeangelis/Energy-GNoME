from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DB = ["cathodes", "perovskites", "thermoelectrics"]
DB_LINK = [
    "https://paolodeangelis.github.io/Energy-GNoME/apps/cathodes/dashboard.html",
    "https://paolodeangelis.github.io/Energy-GNoME/apps/perovskites/dashboard.html",
    "https://paolodeangelis.github.io/Energy-GNoME/apps/thermoelectrics/dashboard.html",
]


root_dir = Path(__file__).parent.parent.parent

out_file = open(root_dir / "docs" / "assets" / "partial" / "dp_cardinality.md", "w")
out_file.write(
    " Material class ".ljust(20)
    + "|"
    + " # Sub-class ".ljust(20)
    + "|"
    + " # Materials (unique) ".ljust(20)
    + "|"
    + " Dashbard "
    "\n"
)
out_file.write("-" * 20 + "|" + "-" * 20 + "|" + "-" * 20 + "|" + "-" * 40 + "\n")

for db_name, db_link in zip(DB, DB_LINK):
    dbs_paths = list((root_dir / "data" / "final" / db_name).rglob("candidates.[jJ][sS][oO][nN]"))
    m_id = []
    n_sub = len(dbs_paths)
    for db_path in dbs_paths:
        m_id += pd.read_json(db_path)["Material Id"].tolist()
    n_unique = len(np.unique(m_id))
    out_file.write(
        f" {db_name.capitalize()} ".ljust(20)
        + "|"
        + f" {n_sub} ".ljust(20)
        + "|"
        + f" {n_unique} ".ljust(20)
        + "|"
        + f"[:octicons-arrow-right-24: Explore {db_name.capitalize()} :material-database:]({db_link}) "
        + "\n"
    )


out_file.write("\n" + "[^1]: Last update: {}".format(datetime.today().strftime("%d/%m/%Y %H:%M:%S")) + "\n")

out_file.close()
