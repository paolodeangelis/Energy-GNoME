# Perovskite Materials: Batabase Preparation Pipeline

This notebook demonstrates how to prepare the database for studying **perovskite materials** using the `energy_gnome` library.

The pipeline covers:

1. Retrieval and processing of perovskite data
2. Removal of overlaps from a background MP database
3. Database cleaning, filtering, and `CIF` file management

This workflow lays the foundation for ML-based screening or property prediction tasks involving perovskites.


```python
from energy_gnome.dataset import MPDatabase, PerovskiteDatabase
from pathlib import Path

# Change data_dir to reflect your project's folder structure.
# Here, we assume that there are a `notebook` and a `data`
# subfolder in the main project folder.
data_dir = Path(".").resolve().parent / "data"
```


## Database Preparation
### Perovskite Materials

We begin by initializing a `PerovskiteDatabase`.

- `name`: Defines a unique name for this database instance. Use distinct names for different projects or dataset versions to avoid accidental overwriting.
- `data_dir`: Sets the root directory where all files will be stored (e.g., raw and processed datasets, CIFs).
- `allow_raw_update()`: Enables updates to the raw data stage, allowing newly retrieved entries to be stored.


```python
perov_db = PerovskiteDatabase(name="perovskites", data_dir=data_dir)
perov_db.allow_raw_update()

print(perov_db)
```


Retrieves perovskite materials. Returns both the raw materials and associated metadata.

For more information about how to query from MP, refer to the `mp_querying.ipynb` notebook.


```python
db, materials = perov_db.retrieve_materials(mute_progress_bars=False)
```


Compares newly retrieved entries with the existing raw database and updates if changes are found.


```python
perov_db.compare_and_update(new_db=db, stage="raw")
```


Saves the raw CIF files and database records to disk.


```python
perov_db.save_cif_files(
    stage="raw",
    mute_progress_bars=False,
)

perov_db.save_database(stage="raw")
```


Processes the perovskite data by:

- Filtering based on v band gap $\left(0.0 < E_g ≤ 2.5 \mathrm{eV}\right)$ for PV applications (customizable)
- Removing magnetic materials (optional)
- Removing metallic materials

CIF files are saved for downstream use.


```python
perov_db.process_database(
    band_gap_lower=0.0, band_gap_upper=2.5, inplace=True, clean_magnetic=True
)
perov_db.copy_cif_files(stage="processed")
```


### MP Database

We now prepare a background database using MP materials, excluding known perovskite structures.


Initializes the MP database.


```python
mp_db = MPDatabase(name="mp_no_perov", data_dir=data_dir)
mp_db.allow_raw_update()

print(mp_db)
```


Removes materials from MP that overlap with the raw perovskite set.

This avoids contamination or label leakage in later ML tasks.


```python
mp_database = mp_db.remove_cross_overlap("raw", perov_db.get_database("raw"))
```


Processes the filtered MP dataset by:

- Applying a custom band gap range
- Retaining magnetic materials (optional behavior)

This step uses the **same cleaning logic** as the perovskite pipeline to ensure consistent feature generation.

```python
mp_clean = perov_db.process_database(
    band_gap_lower=0.0,
    band_gap_upper=10.0,
    inplace=False,
    db=mp_database,
    clean_magnetic=False,
)

mp_db.databases["processed"] = mp_clean
mp_db.save_database("processed")
```


Saves CIF files for downstream use.


```python
df = mp_db.get_database(stage="processed")
mp_db.save_cif_files(stage="processed", mute_progress_bars=False)
```
