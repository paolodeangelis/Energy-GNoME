# E3NN models: Regression Workflow

This notebook walks through the process of building a **E3NN regressor** for predicting targeted properties using the `energy_gnome` framework.

The pipeline includes:

1. Data loading and cleaning
2. Balanced train/test splits
3. Feature generation (structure- and composition-based)
4. Model training and evaluation


```python
from energy_gnome.dataset import MPDatabase, PerovskiteDatabase
from energy_gnome.models import E3NNRegressor
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
    target_property="band_gap", valid_size=0.2, test_size=0.05, save_split=True
)
```


??? tip "Optional"

    ```python
    mp_db.split_regressor(
        target_property="band_gap", valid_size=0.2, test_size=0.05, save_split=True
    )
    ```


## Regressor Initialization

Initializes a Euclidian Neural Network (E3NN) regressor.

- `n_committers`: Number of E3NN models trained.
- Uses **structural** and **compositional features** via `Matminer`.


```python
regressor_model = E3NNRegressor(
    model_name="perov_e3nn",
    target_property="band_gap",
    models_dir=models_dir,
    figures_dir=figures_dir,
)
```


```python
regressor_model.set_model_settings(
    n_committers=4,
    l_max=3,
    r_max=5.0,
    conv_layers=2,
    device="cuda:0",  # change according to your device specifications
    batch_size=16,
)
```


## Feature Engineering and Dataloader Initialization

Generates input dataloader objects for model training using structural/compositional representations.

??? tip "Optional"

    If you are training a **mixed** regression model, you should use `databases=[perov_db, mp_db]` when calling `regressor_model.create_dataloader()` in the next cells.


```python
train_dl, n_neigh_mean = regressor_model.create_dataloader(
    databases=[perov_db], subset="training"
)

valid_dl, _ = regressor_model.create_dataloader(
    databases=[perov_db], subset="validation"
)

test_dl, _ = regressor_model.create_dataloader(databases=[perov_db], subset="testing")
```


## Training

Compiles and trains the E3NN regressor.


```python
regressor_model.compile(num_neighbors=n_neigh_mean, scheduler_settings={"gamma": 0.98})

regressor_model.fit(
    dataloader_train=train_dl, dataloader_valid=valid_dl, n_epochs=10, parallelize=False
)
```


## Evaluation

Evaluates the trained model on both training and test splits.


```python
train_predictions = regressor_model.evaluate(dataloader=train_dl)
valid_predictions = regressor_model.evaluate(dataloader=valid_dl)
test_predictions = regressor_model.evaluate(dataloader=test_dl)
```

## Visualization

Produces history plots showing training & validation vs. # of epochs, and parity plots showing predicted vs. true E<sub>g</sub> values.


```python
regressor_model.plot_history()
```


```python
regressor_model.plot_parity(predictions_dict=train_predictions, include_ensemble=True)
```


```python
regressor_model.plot_parity(predictions_dict=valid_predictions, include_ensemble=True)
```


```python
regressor_model.plot_parity(predictions_dict=test_predictions, include_ensemble=True)
```
