# GNoME Database Screening

This notebook demonstrates how to:

1. **Load** and **preprocess** the GNoME database
2. **Remove overlapping** materials between GNoME and other databases
3. Use a pretrained Gradient Boosted Decision Tree (GBDT) classifier to **screen candidate materials**
4. Use a pretrained E(3)NN (can be switched to GBDT) regressor to **predict** band gaps on screened candidates
5. Save and manage prediction results

The pipeline leverages both compositional and structural features, and focuses on efficient screening and regression of materials properties.


```python
from energy_gnome.dataset import GNoMEDatabase, MPDatabase
from energy_gnome.models import E3NNRegressor, GBDTClassifier
from pathlib import Path

# Change data_dir to reflect your project's folder structure.
# Here, we assume that there are a `notebook`, a `data`, a `models`,
# and a `figures` subfolder in the main project folder.
data_dir = Path(".").resolve().parent / "data"
models_dir = Path(".").resolve().parent / "models"
figures_dir = Path(".").resolve().parent / "figures"
```


## Database creation
Load the GNoME database and allow updates to raw data


```python
gnome_db = GNoMEDatabase(name="gnome", data_dir=data_dir)
gnome_db.allow_raw_update()

print(gnome_db)
```


Access the raw dataset from GNoME


```python
gnome_db.get_database("raw")
```


Pulls the raw GNoME materials dataset. As of now, there is still no implementation of the pipeline to download and extract the GNoME database from [Google Deepmind's repository](https://github.com/google-deepmind/materials_discovery).


```python
df = gnome_db.retrieve_materials()
```


Compares the newly retrieved entries with the existing raw dataset and updates if necessary.


```python
gnome_db.compare_and_update(df, "raw")
```


Saves CIF files for downstream use. Since the files come zipped, this may take some time.


```python
gnome_db.save_cif_files()
```


## Background Database

We now prepare a background dataset using MP materials, to exclude known materials from GNoME. This is necessary since MP has started including GNoME-originated materials calculated using `r2SCAN`.


```python
mp_db = MPDatabase(name="mp", data_dir=data_dir)
```


Removes materials from GNoME that overlap with the raw MP set.

This avoids label leakage in later ML tasks.


```python
gnome_db.remove_cross_overlap(stage="raw", df=mp_db.get_database("raw"), save_db=True)

print(gnome_db)
```


## Classification Screening

Loads the trained classifier model and screens GNoME entries for potential perovskites.


```python
classifier_model = GBDTClassifier(
    model_name="perovskites_gbdt", models_dir=models_dir, figures_dir=figures_dir
)

classifier_model.load_trained_models()
```


```python
screened_df = classifier_model.screen(
    db=gnome_db, featurizing_mode="structure", save_processed=True
)
```


## Target Property Prediction

Applies the trained regression model to the positively screened candidates, estimating their $E_g$ values.

This completes the full screening pipeline.


```python
regressor_model = E3NNRegressor(
    model_name="perov_e3nn",
    target_property="band_gap",
    models_dir=models_dir,
    figures_dir=figures_dir,
)

regressor_model.load_trained_models()
```


```python
final_df = regressor_model.predict(gnome_db, confidence_threshold=0.5, save_final=True)
```
