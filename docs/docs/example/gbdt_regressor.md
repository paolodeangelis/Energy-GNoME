# GBDT models: Regression Workflow

This notebook walks through the process of building a **GBDT regressor** for predicting targeted properties using the `energy_gnome` framework.

The pipeline includes:

1. Data loading and cleaning
2. Balanced train/test splits
3. Feature generation (structure- and/or composition-based)
4. Model training and evaluation


```python
from energy_gnome.dataset import MPDatabase, PerovskiteDatabase
from energy_gnome.models import GBDTRegressor
from pathlib import Path

# Change data_dir to reflect your project's folder structure.
# Here, we assume that there are a `notebook`, a `data`, a `models`,
# and a `figures` subfolder in the main project folder.
data_dir = Path(".").resolve().parent / "data"
models_dir = Path(".").resolve().parent / "models"
figures_dir = Path(".").resolve().parent / "figures"
```


## Data preparation

We begin by loading the cleaned database.

**Info**: in some cases, e.g. for the estimation of the band gap (E<sub>g</sub>) of perovskites, you may want to train **mixed** regression models, including non-specialized materials in your training set. In that case, you should also run the cells marked as `optional`.


```python
perov_db = PerovskiteDatabase(name="perovskites", data_dir=data_dir)
print(perov_db)
```


??? tip "Optional"

    ```python
    mp_db = MPDatabase(name="mp_no_perov", data_dir=data_dir)
    print(mp_db)
    ```


Splits the datasets into training/testing subsets, while:
- Balancing class labels
- Ensuring uniform element distribution across splits


```python
perov_db.split_regressor(
    target_property="band_gap", valid_size=0.0, test_size=0.2, save_split=True
)
```


??? tip "Optional"

    ```python
    mp_db.split_regressor(
        target_property="band_gap", valid_size=0.2, test_size=0.05, save_split=True
    )
    ```


## Regressor Initialization

Initializes a Gradient Boosted Decision Tree (GBDT) regressor.

- `n_committers`: Number of GBDT models trained.
- Uses **structural** and/or **compositional features** via `Matminer`.


```python
regressor_model = GBDTRegressor(
    model_name="perov_gbdt",
    target_property="band_gap",
    models_dir=models_dir,
    figures_dir=figures_dir,
)

regressor_model.set_model_settings(n_committers=4)
```


## Feature Engineering

Generates input features for model training using structural/compositional representations.

**Warning!** Only compositional features are available for thermoelectric materials as of this version of `energy_gnome`.

??? tip "Optional"

    If you are training a **mixed** regression model, you should use `databases=[perov_db, mp_db]` when calling `regressor_model.create_dataloader()` in the next cells.


```python
train_feat = regressor_model.featurize_db(databases=perov_db, mode="structure")

test_feat = regressor_model.featurize_db(
    databases=perov_db, subset="testing", mode="structure"
)
```


## Training

Compiles and trains the GBDT regressor.


```python
regressor_model.compile(n_jobs=6, scoring="neg_root_mean_squared_error")

regressor_model.fit(df=train_feat)
```


## Evaluation

Evaluates the trained model on both training and test splits.


```python
regressor_model.load_trained_models()

train_preds = regressor_model.evaluate(df=train_feat)
test_preds = regressor_model.evaluate(df=test_feat)
```


## Visualization

Produces parity plots showing predicted vs. true E<sub>g</sub> values.


```python
regressor_model.plot_parity(predictions_dict=train_preds)
```


```python
regressor_model.plot_parity(predictions_dict=test_preds)
```
