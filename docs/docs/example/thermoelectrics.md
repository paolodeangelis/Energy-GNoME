# Thermoelectric Materials: ML Modeling & Screening Pipeline

This notebook demonstrates how to use the `energy_gnome` library to build and apply machine learning models for predicting thermoelectric performance.

The complete pipeline includes:

1. **Dataset preparation** — from raw retrieval to cleaning and downsampling.
2. **Binary classification** — to identify potential thermoelectric candidates.
3. **Regression modeling** — to predict the thermoelectric figure of merit (ZT).
4. **High-throughput screening** — applying trained models to external material databases.

This workflow reflects real-world ML practices: **data curation &rarr; feature engineering &rarr; model training/testing &rarr; application**.


```python
from energy_gnome.dataset import GNoMEDatabase, MPDatabase, ThermoelectricDatabase
from energy_gnome.models import GBDTClassifier, GBDTRegressor
import numpy as np
from pathlib import Path

# Change data_dir to reflect your project's folder structure.
# Here, we assume that there are a `notebook`, a `data`, a `models`,
# and a `figures` subfolder in the main project folder.
data_dir = Path(".").resolve().parent / "data"
models_dir = Path(".").resolve().parent / "models"
figures_dir = Path(".").resolve().parent / "figures"
```


## Dataset creation
### Thermoelectric Materials

We start by initializing a database of known thermoelectric compounds using `ThermoelectricDatabase`.

- `name`: Defines a unique name for this database instance. Use distinct names for different projects or dataset versions to avoid accidental overwriting.
- `data_dir`: Sets the root directory where all files will be stored (e.g., raw and processed datasets, CIFs).
- `allow_raw_update()`: Enables updates to the raw data stage, allowing newly retrieved entries to be stored.

For initializing other database types, see the respective notebooks.


```python
thermo_db = ThermoelectricDatabase(name="thermoelectrics", data_dir=data_dir)

thermo_db.allow_raw_update()
```


Pulls the raw thermoelectric materials dataset. Here, we assume that the estm.xlsx file is already present in the `data/external/<name>` folder.


```python
raw_thermo = thermo_db.retrieve_materials()
```


Compares the newly retrieved entries with the existing raw dataset and updates if necessary.


```python
thermo_db.compare_and_update(new_db=raw_thermo, stage="raw")
```


Processes the dataset: cleaning, standardizing, and generating compositional features (in-place).


```python
thermo_db.process_database(inplace=True)
```


### Materials Project (MP) Database

We initialize a generic MP-based database to act as a background (non-thermoelectric) dataset.

This notebook assumes the MP dataset has already been downloaded. For retrieval instructions, see the MP-specific notebook.

Because the pipeline is strictly compositional, structure files (CIFs) from MP are not required here.


```python
mp_db = MPDatabase(name="mp_no_thermo", data_dir=data_dir)

mp_db.allow_raw_update()
```


Returns the cleaned, processed thermoelectric dataset.


```python
thermo_proc = thermo_db.get_database(stage="processed")
```


Removes overlapping materials between the MP and thermoelectric datasets to prevent data leakage in classification.


```python
mp_db.remove_cross_overlap(stage="raw", df=thermo_proc, save_db=True)
```


Randomly downsamples the MP dataset to approximately match the thermoelectric set's size.


```python
mp_df_red = mp_db.random_downsample(
    size=round(len(thermo_proc["formula_pretty"].unique()) * 1.1),
    new_name="mp_no_thermo_red",
    stage="raw",
)
```


Reinitializes the downsampled MP dataset.


```python
mp_db_red = MPDatabase(name="mp_no_thermo_red", data_dir=data_dir)

print(mp_db_red)
```


Processes the reduced MP dataset using the same featurization logic as the thermoelectric database.
Ensures feature consistency across datasets.


```python
mp_clean = thermo_db.process_database(
    inplace=False, df=mp_df_red, temp_list=list(np.arange(300, 1000, 130, float))
)

mp_db_red.databases["processed"] = mp_clean

mp_db_red.save_database(stage="processed")
```


## Binary Classification — GBDT model
### Data preparation

We begin by loading the cleaned databases.


```python
thermo_db = ThermoelectricDatabase(name="thermoelectrics", data_dir=data_dir)
print(thermo_db)
```


```python
mp_db = MPDatabase(name="mp_no_thermo_red", data_dir=data_dir)
print(mp_db)
```


Splits the datasets into training/testing subsets, while:
- Balancing class labels
- Ensuring uniform element distribution across splits


```python
thermo_db.split_classifier(test_size=0.2, balance_composition=True, save_split=True)

mp_db.split_classifier(test_size=0.2, balance_composition=True, save_split=True)
```


### Classifier Initialization

Initializes a Gradient Boosted Decision Tree (GBDT) classifier.

- `n_committers`: Number of GBDT models trained.
- Uses only **compositional features** via `Matminer`.


```python
classifier_model = GBDTClassifier(model_name="thermo_gbdt", models_dir=models_dir)

classifier_model.set_model_settings(n_committers=10)
```


### Feature Engineering

Generates input features for model training using compositional representations.


```python
train_feat = classifier_model.featurize_db(
    databases=[thermo_db, mp_db], mode="composition"
)

test_feat = classifier_model.featurize_db(
    databases=[thermo_db, mp_db], subset="testing", mode="composition"
)
```


### Training

Compiles and trains the GBDT classifier.


```python
classifier_model.compile(n_jobs=6)

classifier_model.fit(df=train_feat)
```


### Evaluation

Evaluates the trained model on both training and test splits.


```python
classifier_model.load_trained_models()

train_preds = classifier_model.evaluate(df=train_feat)
test_preds = classifier_model.evaluate(df=test_feat)
```


### Visualization

Plots classification performance:
- ROC curve with AUC (Area Under the Curve)
- Precision-Recall curve
- Recall-Threshold curve


```python
classifier_model.plot_performance(predictions_dict=train_preds)
```


```python
classifier_model.plot_performance(predictions_dict=test_preds)
```


## Regression - GBDT model
### Data preparation

We set up the regression task using the thermoelectric dataset only.


```python
thermo_db = ThermoelectricDatabase(name="thermoelectrics", data_dir=data_dir)
```


Splits the dataset for regression on the `ZT` target.
- No validation set is used in the GBDT models.


```python
thermo_db.split_regressor(
    target_property="ZT", valid_size=0.0, test_size=0.2, save_split=True
)
```


### Regressor Initialization

Initializes a GBDT-based regressor for predicting the thermoelectric figure of merit (ZT).


```python
regressor_model = GBDTRegressor(
    model_name="thermo_gbdt", target_property="ZT", models_dir=models_dir
)

regressor_model.set_model_settings(n_committers=4)
```


### Feature Engineering

Generates training and testing features from compositions.


```python
train_feat = regressor_model.featurize_db(databases=[thermo_db], mode="composition")

test_feat = regressor_model.featurize_db(
    databases=[thermo_db], subset="testing", mode="composition"
)
```


### Training

Compiles and fits the model.
- Uses **RMSE** as the scoring metric.


```python
regressor_model.compile(n_jobs=6, scoring="neg_root_mean_squared_error")

regressor_model.fit(df=train_feat)
```


### Evaluation

Evaluates model predictions on both training and testing data.


```python
regressor_model.load_trained_models()

train_preds = regressor_model.evaluate(df=train_feat)
test_preds = regressor_model.evaluate(df=test_feat)
```


### Visualization

Produces parity plots showing predicted vs. true ZT values.


```python
regressor_model.plot_parity(predictions_dict=train_preds)
```


```python
regressor_model.plot_parity(predictions_dict=test_preds)
```


## GNoME screening
### GNoME Database Initialization

Initializes a database of GNoME-generated materials (novel candidates).

Assumes the raw GNoME dataset is already available. See the GNoME-specific notebook for retrieval instructions.


```python
gnome_db = GNoMEDatabase(name="gnome", data_dir=data_dir)
print(gnome_db)
```


### Pre-Screening Cleanup

Removes any material entries from GNoME that overlap with:
- MP dataset
- Thermoelectric dataset

This avoids label leakage in later ML tasks.


```python
gnome_db.remove_cross_overlap(stage="raw", df=mp_db.get_database("raw"), save_db=True)
```


```python
gnome_db.remove_cross_overlap(
    stage="raw", df=thermo_db.get_database("processed"), save_db=True
)
```


### Processing Pipeline

Processes the GNoME dataset using the same cleaning/featurization logic used earlier.


```python
raw_df = gnome_db.get_database(stage="raw")

thermo_db = ThermoelectricDatabase(name="thermoelectrics", data_dir=data_dir)

gnome_clean = thermo_db.process_database(inplace=False, df=raw_df)

gnome_db.databases["raw"] = gnome_clean
gnome_db.save_database(stage="raw")
```


### Classification Screening

Loads the trained classifier model and screens GNoME entries for potential thermoelectrics.


```python
classifier_model = GBDTClassifier(
    model_name="thermo_gbdt", models_dir=models_dir, figures_dir=figures_dir
)

classifier_model.load_trained_models()
```


```python
screened_df = classifier_model.screen(
    db=gnome_db, featurizing_mode="composition", save_processed=True
)
```


### ZT Prediction

Applies the trained regression model to the positively screened candidates, estimating their ZT values.

This completes the full screening pipeline.


```python
regressor_model = GBDTRegressor(
    model_name="thermo_gbdt", target_property="ZT", models_dir=models_dir
)
```


```python
df_final = regressor_model.predict(
    db=gnome_db, confidence_threshold=0.5, featurizing_mode="composition"
)
```
