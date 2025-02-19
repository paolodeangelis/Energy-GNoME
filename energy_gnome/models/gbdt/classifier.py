import glob
import json
import os
from pathlib import Path
import pickle
import time
import warnings

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import PCG64, Generator
import pandas as pd

# from mat2vec.processing import MaterialsTextProcessor
from pymatgen.analysis import magnetism
from pymatgen.core import Composition, Structure
import sklearn
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
)
from sklearn.metrics import (
    auc,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import yaml

from energy_gnome.config import DEFAULT_GBDT_SETTINGS, FIGURES_DIR, MODELS_DIR
from energy_gnome.dataset.base_dataset import BaseDatabase
from energy_gnome.dataset.gnome import GNoMEDatabase
from energy_gnome.models.abc_model import BaseClassifier
from energy_gnome.utils.readers import load_json, load_yaml, save_yaml

from .utils import featurizing_structure_pipeline

BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"
tqdm.pandas(bar_format=BAR_FORMAT)


class GBDTClassifier(BaseClassifier):
    def __init__(
        self,
        model_name: str,
        models_dir: Path | str = MODELS_DIR,
        figures_dir: Path | str = FIGURES_DIR,
    ):
        """
        Initialize the GBDTClassifier with root data and models directories.

        Sets up the gbdt regressor structure for building and training the regressor models.

        Args:
            model_name (str): ...
            models_dir (Path, optional): Root directory path for storing trained models.
                                         Defaults to MODELS_DIR from config.
        """
        super().__init__(model_name=model_name, models_dir=models_dir, figures_dir=figures_dir)

    def _find_model_states(self):
        models_states = []
        if any(self.models_dir.iterdir()):
            models_states = [f_ for f_ in self.models_dir.iterdir() if f_.match("*.pkl")]
        return models_states

    def set_model_settings(self, yaml_file: Path | str | None = None, **kargs):
        # Accessing model settings (YAML FILE)
        if yaml_file:
            self._load_model_setting(yaml_file)

        # Accessing model settings
        for att, defvalue in DEFAULT_GBDT_SETTINGS.items():
            if att in kargs:
                setattr(self, att, kargs[att])
            else:
                try:
                    att_exist = getattr(self, att)
                    att_exist = att_exist == att_exist
                except AttributeError:
                    att_exist = False
                if not att_exist:
                    setattr(self, att, defvalue)
                    logger.warning(f"using default value {defvalue} for {att} setting")

        self._model_spec = (
            "model.gbdt_classifier"  # + "." + time.strftime("%y%m%d", time.localtime())
        )
        if yaml_file is None or os.path.dirname(str(yaml_file)) != str(self.models_dir):
            self._save_model_settings()

    def _save_model_settings(self):
        settigns_path = self.models_dir / (self._model_spec + ".yaml")
        settings = {}
        for att, _ in DEFAULT_GBDT_SETTINGS.items():
            settings[att] = getattr(self, att)
        logger.info(f"saving models 'general' settings in {settigns_path}")
        save_yaml(settings, settigns_path)

    def _load_model_setting(self, yaml_path):
        settings = load_yaml(yaml_path)
        for att, _ in DEFAULT_GBDT_SETTINGS.items():
            setattr(self, att, settings[att])

    '''
    def featurizer(
        self,
        databases: list[BaseDatabase],
        subset: str,
        max_dof: int = None,
        mute_warnings: bool = True
        ) -> pd.DataFrame:
        """
        Load and featurize the specified databases.

        Checks for the presence of an existing database file for the given stage
        and loads it into a pandas DataFrame. If the database file does not exist,
        logs a warning and returns an empty DataFrame.
        Builds a pandas DataFrame ready for model training.

        Args:
            dataset (pd.DataFrame) = ...

        Returns:
            pd.DataFrame: The built database or an empty DataFrame if not found.
        """
        if not isinstance(databases, list):
            databases=[databases]

        if mute_warnings:
            logger.warning("Warnings from third-party libraries are disabled. To enable them, set 'mute_warnings' = False")
            self.max_dof = max_dof

        property_to_add = {
            "formula": "str",
            "composition": "object",
            "structure": "object",
            "is_specialized": "float",
        }

        with warnings.catch_warnings():
            if mute_warnings:
                warnings.simplefilter("ignore")
            db_list = []
            for db in databases:
                db_list.append(db.load_classifier_data(subset))

            dataset = pd.concat(db_list, ignore_index=True)

            db = pd.DataFrame()
            for column, dtype in property_to_add.items():
                db[column] = pd.Series(dtype=dtype)

            dataset.reset_index(drop=True, inplace=True)

            for j in range(len(dataset)):
                db.at[j, "is_specialized"] = dataset.at[j, "is_specialized"].astype("float")
                db.at[j, "composition"] = Composition(dataset.at[j, "formula_pretty"])
                db.at[j, "structure"] = Structure.from_file(dataset.at[j, "cif_path"])
                db.at[j, "formula"] = db.at[j, "composition"].reduced_formula

            datasets_class = db.copy()

            db_feature = featurizing_structure_pipeline(datasets_class)
            db_feature["is_specialized"] = datasets_class.set_index("formula")["is_specialized"]
            db_feature.dropna(axis=0, how="any", inplace=True)
            db_feature = db_feature.select_dtypes(exclude=["object", "bool"])

        database_class = db_feature.copy()

        if max_dof is None:
            n_items, n_features = database_class.shape
            self.max_dof = min(n_items // 10, n_features - 1)

        n_is_specialized = database_class["is_specialized"].value_counts().get(1.0, 0)
        n_is_not_specialized = database_class["is_specialized"].value_counts().get(0.0, 0)
        logger.debug(f"number of specialized examples: {n_is_specialized}")
        logger.debug(f"number of non-specialized examples: {n_is_not_specialized}")

        return database_class
    '''

    def featurize_db(
        self,
        databases: list[BaseDatabase],
        subset: str | None = None,
        max_dof: int = None,
        mute_warnings: bool = True,
    ) -> pd.DataFrame:
        """
        Load and featurize the specified databases efficiently.
        """

        if not isinstance(databases, list):
            databases = [databases]

        if mute_warnings:
            logger.warning(
                "Third-party warnings disabled. Set 'mute_warnings=False' to enable them."
            )
            self.max_dof = max_dof

        if isinstance(databases[0], GNoMEDatabase):
            dataset = databases[0].get_database("raw")
        else:
            # Load all database subsets efficiently
            dataset = pd.concat(
                [db.load_classifier_data(subset) for db in databases], ignore_index=True
            )

        if dataset.empty:
            logger.warning("No data loaded for featurization.")
            return pd.DataFrame()

        with warnings.catch_warnings():
            if mute_warnings:
                warnings.simplefilter("ignore")

            # Apply batch processing for transformations
            tqdm.pandas(desc="Processing structures")
            if isinstance(databases[0], GNoMEDatabase):
                dataset["composition"] = dataset["Reduced Formula"].progress_apply(Composition)
            else:
                dataset["composition"] = dataset["formula_pretty"].progress_apply(Composition)
            dataset["structure"] = dataset["cif_path"].progress_apply(
                lambda x: Structure.from_file(x)
            )
            dataset["formula"] = dataset["composition"].apply(lambda x: x.reduced_formula)
            dataset["is_specialized"] = dataset["is_specialized"].astype(float)

            # Run the featurization pipeline
            if isinstance(databases[0], GNoMEDatabase):
                db_feature = featurizing_structure_pipeline(dataset)
            else:
                db_feature = featurizing_structure_pipeline(dataset)

            # Retain only numerical features
            db_feature = db_feature.select_dtypes(exclude=["object", "bool"])
            db_feature["is_specialized"] = dataset.set_index("formula")["is_specialized"]
            db_feature.dropna(inplace=True)  # Remove NaN rows

            # Adjust max_dof if needed
            if max_dof is None:
                n_items, n_features = db_feature.shape
                self.max_dof = min(n_items // 10, n_features - 1)

        if not isinstance(databases[0], GNoMEDatabase):
            # Log class distribution
            specialized_counts = db_feature["is_specialized"].value_counts()
            logger.debug(f"Number of specialized examples: {specialized_counts.get(1.0, 0)}")
            logger.debug(f"Number of non-specialized examples: {specialized_counts.get(0.0, 0)}")

        return db_feature

    def build_classifier(self, n_jobs: int = 1, max_dof: int = None):
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("feature_selector", RFE(estimator=GradientBoostingClassifier(), step=25)),
                ("classifier", GradientBoostingClassifier()),
            ],
        )
        param_grid = {
            "classifier__n_estimators": [50, 100, 250, 500],
            "feature_selector__n_features_to_select": [int(max_dof * 0.5), int(max_dof * 1)],
        }

        stratified_kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
        search = GridSearchCV(
            pipe, param_grid, n_jobs=n_jobs, verbose=2, cv=stratified_kfold, scoring="roc_auc"
        )

        return search

    def compile(
        self,
        n_jobs: int = 1,
    ):

        # logger.info("[STEP 2] Featurize and format subsets for training")
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     training_db_list = []
        #     for db in databases:
        #         training_db_list.append(db.load_classifier_data("training"))

        #     training_db = pd.concat(training_db_list, ignore_index=True)

        #     train_feat = self.db_featurizer(training_db)

        # if max_dof is None:
        #     n_items, n_features = train_feat.shape
        #     max_dof = min(n_items // 10, n_features - 1)

        # n_is_specialized = train_feat["is_specialized"].value_counts().get(1.0, 0)
        # n_is_not_specialized = train_feat["is_specialized"].value_counts().get(0.0, 0)
        # logger.debug(f"number of specialized examples: {n_is_specialized}")
        # logger.debug(f"number of non-specialized examples: {n_is_not_specialized}")

        logger.info("Build classifiers")
        self.search = self.build_classifier(n_jobs=n_jobs, max_dof=self.max_dof)

    def fit(self, df: pd.DataFrame):
        models = {}
        for i in tqdm(range(self.n_committers), desc="training models"):
            models[f"model_{i}"] = self.search.fit(df.iloc[:, :-1], df.iloc[:, -1])

        for i in tqdm(range(self.n_committers), desc="saving"):
            model_path = self.models_dir / (self._model_spec + f".rep{i}.pkl")
            model_ = models[f"model_{i}"]
            with open(model_path, "wb") as file_:
                pickle.dump(model_, file_)

    def load_trained_models(self):
        for f in os.listdir(self.models_dir):
            if f.endswith(".yaml"):
                yaml_path = self.models_dir / f
                self.set_model_settings(yaml_file=yaml_path)
        i = 0
        for f in os.listdir(self.models_dir):
            if f.endswith(".pkl"):
                model_history_path = self.models_dir / f
                logger.info(f"Loading model with weights in {model_history_path}")
                with open(model_history_path, "rb") as file_:
                    self.models[f"model_{i}"] = pickle.load(file_)
                self.models[f"model_{i}"].pool = True
                i += 1

        return [f for f in os.listdir(self.models_dir) if f.endswith(".pkl")]

    def evaluate(self, df: pd.DataFrame, return_df: bool = False):
        if return_df:
            predictions = pd.DataFrame(columns=["true"])
            predictions["true"] = df[self.target_property]

            for i in tqdm(range(self.n_committers), desc="models"):
                predictions[f"model_{i}"] = np.empty((len(df), 1)).tolist()
                predictions[f"model_{i}"] = self.models[f"model_{i}"].predict_proba(
                    df.iloc[:, :-1]
                )[:, 1]

        else:
            predictions = {}
            for i in tqdm(range(self.n_committers), desc="models"):
                predictions[f"model_{i}"] = pd.DataFrame()
                predictions[f"model_{i}"]["true_value"] = df[self.target_property]
                predictions[f"model_{i}"]["prediction"] = self.models[f"model_{i}"].predict_proba(
                    df.iloc[:, :-1]
                )[:, 1]

        return predictions

    def plot_performance(
        self, predictions_dict: dict[str, pd.DataFrame], include_ensemble: bool = True
    ):
        all_predictions = []
        colors = plt.cm.tab10.colors

        # Create a single figure and a grid layout
        fig = plt.figure(figsize=(7, 2), dpi=150)
        grid = fig.add_gridspec(1, 3, wspace=0.5, hspace=0.07)
        ax = [fig.add_subplot(grid[j]) for j in range(3)]

        for i, (model, data) in enumerate(predictions_dict.items()):
            y_true = data["true_value"].values  # Ensure correct format
            y_predictions = data["prediction"].values

            if include_ensemble:
                all_predictions.append(y_predictions)

            fpr, tpr, _ = roc_curve(y_true, y_predictions)
            precision, recall, thresholds = precision_recall_curve(y_true, y_predictions)

            # Use colors from plt.cm.tab10
            color = colors[i % len(colors)]

            ax[0].plot(fpr, tpr, label=f"{model} (AUC: {auc(fpr, tpr):.2f})", color=color)
            ax[1].plot(thresholds, precision[:-1], label=f"{model}", color=color)
            ax[2].plot(thresholds, recall[:-1], label=f"{model}", color=color)

        # If ensemble is enabled, add an extra curve
        if include_ensemble and all_predictions:
            y_ensemble = np.mean(np.column_stack(all_predictions), axis=1)
            fpr, tpr, _ = roc_curve(y_true, y_ensemble)
            precision, recall, thresholds = precision_recall_curve(y_true, y_ensemble)

            ax[0].plot(fpr, tpr, label="Ensemble", color="black", linestyle="--")
            ax[1].plot(thresholds, precision[:-1], label="Ensemble", color="black", linestyle="--")
            ax[2].plot(thresholds, recall[:-1], label="Ensemble", color="black", linestyle="--")

        # Add diagonal reference line in ROC curve
        ax[0].axline([0.5, 0.5], slope=1, lw=0.85, ls="--", color="k", zorder=2)
        ax[0].set_xlabel("FPR")
        ax[0].set_ylabel("TPR")
        ax[0].set_title("ROC Curve")
        ax[0].set_aspect("equal")

        ax[1].set_xlabel("Threshold")
        ax[1].set_ylabel("Precision")
        ax[1].set_title("Precision Curve")
        ax[1].set_ylim([-0.05, 1.05])

        ax[2].set_xlabel("Threshold")
        ax[2].set_ylabel("Recall")
        ax[2].set_title("Recall Curve")
        ax[2].set_ylim([-0.05, 1.05])

        # Move legend outside the plot
        ax[2].legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=8, borderaxespad=0.0)

        fig.suptitle("Model Performance")
        fig.subplots_adjust(top=0.8, right=0.75)  # Adjust right margin for legend space

        # Save figure
        fig.savefig(self.figures_dir / (self._model_spec + ".png"), dpi=330, bbox_inches="tight")
        fig.savefig(self.figures_dir / (self._model_spec + ".pdf"), dpi=330, bbox_inches="tight")

        plt.show()
