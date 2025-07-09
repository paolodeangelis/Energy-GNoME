# Generic MP Querying Overview

This notebook demonstrates how to use the `energy_gnome` library to query the Materials Project (MP) database via its API.

You will learn how to:

1. Initialize and configure a generic materials database.
2. Retrieve entries using the `MPRester` from `mp_api`.
3. Manage and update local raw datasets.
4. Optionally save structure files (`CIF`) for future use.

\> This workflow is also compatible with specialized subclasses such as `PerovskiteDatabase` and `CathodeDatabase`. Refer to the respective notebooks for targeted use cases.


```python
from energy_gnome.dataset import MPDatabase
from pathlib import Path

# Change data_dir to reflect your project's folder structure.
# Here, we assume that there are a `notebook` and a `data` subfolder
# in the main project folder.
main_dir = Path(".").resolve().parent
data_dir = main_dir / "data"
```

## Dataset creation
### Database Initialization

We begin by initializing a generic MP-based database using the `MPDatabase` class.

- `name`: Defines a unique name for this database instance. Use distinct names for different projects or dataset versions to avoid accidental overwriting.
- `data_dir`: Sets the root directory where all files will be stored (e.g., raw and processed datasets, CIFs).
- `allow_raw_update()`: Enables updates to the raw data stage, allowing newly retrieved entries to be stored.

\> For initializing other MP-based database types, such as `PerovskiteDatabase` or `CathodeDatabase`, consult the respective example notebooks.

```python
mp_db = MPDatabase(name="mp", data_dir=data_dir)
mp_db.allow_raw_update()

print(mp_db)
```


### Data Retrieval

This step fetches material entries from the Materials Project via its API.

To proceed, you **must** have an MP API key. Follow these steps:

1. Register on the [Materials Project](https://next-gen.materialsproject.org).
2. Copy your API key from [here](https://next-gen.materialsproject.org/dashboard).
3. Save it to a `config.yaml` file in your working directory using this format:

```yaml
MP: "<your-api-key>"
```

```python
db, materials = mp_db.retrieve_materials(max_framework_size=6, mute_progress_bars=False)
```

### Updating the Raw Database

Once the data is retrieved, we compare it with the existing raw dataset (if any) and update accordingly.

This ensures:

- New materials are added.
- Existing entries are not duplicated.
- Data integrity is maintained across multiple runs.

```python
mp_db.compare_and_update(new_db=db, stage="raw")
```

### Saving CIF Files

You may optionally save structure files (`CIF` format) for the retrieved materials.

> **_IMPORTANT:_**
To save time and disk space, it’s often more efficient to skip saving CIFs at the raw stage — especially if you plan to downsample or filter the dataset later.

Instead, consider saving CIFs only after:

- The dataset has been processed or cleaned.
- You've finalized the materials you plan to use in model training or screening.

This can significantly reduce IO overhead and file clutter, particularly for large-scale MP queries.

```python
mp_db.save_cif_files(stage="raw", mute_progress_bars=False)
mp_db.save_database("raw")
```
