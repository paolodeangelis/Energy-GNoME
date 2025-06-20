# GBDT models: Binary Classification Workflow

This notebook walks through the process of building a **GBDT classifier** for identifying specialized materials with targeted properties using the `energy_gnome` framework.

The pipeline includes:

1. Data loading and cleaning
2. Balanced train/test splits
3. Feature generation (structure- or composition-based)
4. Model training and evaluation


```python
from energy_gnome.dataset import MPDatabase, PerovskiteDatabase
from energy_gnome.models import GBDTClassifier
from pathlib import Path

# Change data_dir to reflect your project's folder structure.
# Here, we assume that there are a `notebook`, a `data`, and a `models`
# subfolder in the main project folder.
data_dir = Path(".").resolve().parent / "data"
models_dir = Path(".").resolve().parent / "models"
figures_dir = Path(".").resolve().parent / "figures"
```


## Data Preparation

We begin by loading the pre-processed databases:

- `perovskites`: Labeled materials with known properties
- `mp_no_perov`: Generic materials, excluding known perovskites (used as background)

These datasets should have been pre-cleaned and processed using the pipeline in the earlier notebook.


```python
perov_db = PerovskiteDatabase(name="perovskites", data_dir=data_dir)
print(perov_db)
```


```python
mp_db = MPDatabase(name="mp_no_perov", data_dir=data_dir)
print(mp_db)
```


Splits the datasets into training and testing subsets.

The splitting procedure:
- Balances the class labels
- Ensures similar elemental distribution across train/test


```python
perov_db.split_classifier(test_size=0.2, balance_composition=True, save_split=True)

mp_db.split_classifier(test_size=0.2, balance_composition=True, save_split=True)
```


## Classifier Initialization

Initializes a Gradient Boosted Decision Tree (GBDT) classifier.

- `n_committers`: Number of GBDT models trained.
- Uses either **structural** or **compositional features** via `Matminer`.


```python
classifier_model = GBDTClassifier(
    model_name="perov_gbdt", models_dir=models_dir, figures_dir=figures_dir
)

classifier_model.set_model_settings(n_committers=10)
```


## Feature Engineering

Generates input features for model training using structural/compositional representations.

**Warning!** Only compositional features are available for thermoelectric materials as of this version of `energy_gnome`.


```python
train_feat = classifier_model.featurize_db(
    databases=[perov_db, mp_db], mode="structure"
)

test_feat = classifier_model.featurize_db(
    databases=[perov_db, mp_db], subset="testing", mode="structure"
)
```


## Training

Compiles and trains the GBDT classifier.



```python
classifier_model.compile(n_jobs=6)

classifier_model.fit(df=train_feat)
```


## Evaluation

Evaluates the trained model on both training and test splits.


```python
classifier_model.load_trained_models()

train_preds = classifier_model.evaluate(df=train_feat)
test_preds = classifier_model.evaluate(df=test_feat)
```


## Visualization

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
