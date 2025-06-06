import os
from pathlib import Path
import pickle
from typing import Any
import warnings

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Structure
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from energy_gnome.config import DEFAULT_GBDT_SETTINGS, FIGURES_DIR, MODELS_DIR
from energy_gnome.dataset.base_dataset import BaseDatabase
from energy_gnome.dataset.gnome import GNoMEDatabase
from energy_gnome.models.abc_model import BaseRegressor
from energy_gnome.utils.readers import load_yaml, save_yaml

from .utils import featurizing_composition_pipeline, featurizing_structure_pipeline

BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"
tqdm.pandas(bar_format=BAR_FORMAT)


class GBDTRegressor(BaseRegressor):
    def __init__(
        self,
        model_name: str,
        target_property: str,
        models_dir: Path | str = MODELS_DIR,
        figures_dir: Path | str = FIGURES_DIR,
    ):
        """
        Initialize the GBDTRegressor with directories for storing models and figures.

        This class extends `BaseRegressor` to implement a gradient boosted decision tree (GBDT)
        for regression tasks. It sets up the necessary directory structure and configurations
        for training models.

        Args:
            model_name (str): Name of the model, used to create subdirectories.
            target_property (str): The target property the model is trained to predict.
            models_dir (Path | str, optional): Directory for storing trained model weights.
                Defaults to MODELS_DIR from config.
            figures_dir (Path | str, optional): Directory for saving figures and visualizations.
                Defaults to FIGURES_DIR from config.

        Attributes:
            _model_spec (str): Specification string used for model identification.
        """
        self._model_spec = "model.gbdt_regressor." + target_property
        super().__init__(
            model_name=model_name, target_property=target_property, models_dir=models_dir, figures_dir=figures_dir
        )

    def _find_model_states(self):
        models_states = []
        if any(self.models_dir.iterdir()):
            models_states = [f_ for f_ in self.models_dir.iterdir() if f_.match("*.pkl")]
        return models_states

    def load_trained_models(self) -> list[str]:
        """
        Load trained models from the model directory.

        This method searches for trained models by:
        1. Loading model settings from `.yaml` files matching `_model_spec`.
        2. Loading model weights from `.pkl` files.
        3. Storing the loaded models in `self.models`.

        Returns:
            list[str]: A list of `.pkl` model filenames that were found in the directory.
        """
        # Load model settings from YAML files matching the model spec
        for yaml_path in self.models_dir.glob(f"*{self._model_spec}*.yaml"):
            self.set_model_settings(yaml_file=yaml_path)

        i = 0
        loaded_models = []
        for model_path in self.models_dir.glob("*.pkl"):
            try:
                logger.info(f"Loading model from {model_path}")
                with open(model_path, "rb") as file_:
                    model = pickle.load(file_)
                    model.pool = True
                    self.models[f"model_{i}"] = model
                    loaded_models.append(model_path.name)
                    i += 1
            except Exception as e:
                logger.error(f"Error loading model {model_path.name}: {e}")

        return loaded_models

    def _load_model_setting(self, yaml_path):
        """
        Load model settings from a YAML file and assign corresponding attributes.

        This method loads settings from the specified YAML file and sets model attributes
        based on the values in `DEFAULT_GBDT_SETTINGS`. If an attribute is missing, a KeyError
        will be raised.

        Args:
            yaml_path (Path): Path to the YAML file containing the model settings.
        """
        settings = load_yaml(yaml_path)
        for att, _ in DEFAULT_GBDT_SETTINGS.items():
            setattr(self, att, settings[att])

    def _save_model_settings(self):
        """
        Save the current model settings to a YAML file.

        This method saves the current values of the model's attributes (based on `DEFAULT_GBDT_SETTINGS`)
        to a YAML file in the models directory.

        The file is named based on the model specification (`self._model_spec`) with a `.yaml` extension.

        Raises:
            IOError: If the saving process fails.
        """
        settigns_path = self.models_dir / (self._model_spec + ".yaml")
        settings = {}
        for att, _ in DEFAULT_GBDT_SETTINGS.items():
            settings[att] = getattr(self, att)
        logger.info(f"saving models 'general' settings in {settigns_path}")
        save_yaml(settings, settigns_path)

    def set_model_settings(self, yaml_file: Path | str | None = None, **kargs):
        """
        Set model settings either from a YAML file or provided keyword arguments.

        This method allows setting model settings from multiple sources:
        1. If a `yaml_file` is provided, it loads the settings from that file.
        2. If additional settings are provided as keyword arguments (`kargs`), they overwrite
        the default or loaded settings.

        Args:
            yaml_file (Path, str, optional): Path to the YAML file containing the model settings.
            kargs (dict, optional): Dictionary of model settings to override the default ones.

        """
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

        if yaml_file is None or os.path.dirname(str(yaml_file)) != str(self.models_dir):
            self._save_model_settings()

    def featurize_db(
        self,
        databases: list[BaseDatabase],
        subset: str = "training",
        mode: str = "structure",
        max_dof: int = None,
        confidence_threshold: float = 0.5,
        mute_warnings: bool = True,
    ) -> pd.DataFrame:
        """
        Load and featurize materials data from one or more databases using structure or composition-based pipelines.

        This method loads the specified subset of data from a list of `BaseDatabase` objects (e.g., GNoMEDatabase
        or ThermoelectricDatabase) and applies a feature extraction pipeline depending on the selected mode:
        structural or compositional. The output is a clean, numerical feature matrix suitable for training
        machine learning models.

        Args:
            databases (list[BaseDatabase]): One or more databases from which to load materials data.
            subset (str, optional): The subset of the database to load (e.g., "training", "validation").
                Defaults to "training".
            mode (str, optional): Type of featurization to apply. Options:
                - "structure": Parses `.cif` files and applies structure-based features.
                - "composition": Uses only the chemical composition for features.
                Defaults to "structure".
            max_dof (int, optional): Maximum degrees of freedom for downstream model complexity.
                If None, it is computed automatically based on the dataset size.
            confidence_threshold (float, optional): The threshold for filtering out low-confidence entries in the
                `GNoMEDatabase`. Defaults to `0.5`.
            mute_warnings (bool, optional): Whether to suppress warnings during featurization. Defaults to True.

        Returns:
            pd.DataFrame: A processed DataFrame containing only numerical features and the target property.

        Raises:
            ValueError: If `mode` is not one of ["structure", "composition"].

        Notes:
            - For GNoMEDatabase, data is always pulled from the "raw" stage.
            - Structural featurization requires valid CIF paths and may be slower due to file I/O.
            - The target property column is retained alongside features, and rows with missing values are dropped.
            - `tqdm` is used to display progress for long-running operations.
        """

        if not isinstance(databases, list):
            databases = [databases]

        if mute_warnings:
            logger.warning("Third-party warnings disabled. Set 'mute_warnings=False' to enable them.")

        self.max_dof = max_dof

        if isinstance(databases[0], GNoMEDatabase):
            dataset = databases[0].get_database("processed")
            dataset = dataset[dataset["classifier_mean"] > confidence_threshold]
        else:
            # Load all database subsets efficiently
            dataset = pd.concat([db.load_regressor_data(subset) for db in databases], ignore_index=True)

        if dataset.empty:
            logger.warning("No data loaded for featurization.")
            return pd.DataFrame()

        with warnings.catch_warnings():
            if mute_warnings:
                warnings.simplefilter("ignore")

            # Apply batch processing for transformations
            dataset["composition"] = dataset["formula_pretty"].progress_apply(Composition)

            # structure pipeline
            if mode == "structure":
                tqdm.pandas(desc="Processing structures")
                dataset["structure"] = dataset["cif_path"].progress_apply(lambda x: Structure.from_file(x))
                dataset["formula"] = dataset["composition"].apply(lambda x: x.reduced_formula)
                dataset[self.target_property] = dataset[self.target_property].astype(float)

                # Run the featurization pipeline
                if isinstance(databases[0], GNoMEDatabase):
                    db_feature = featurizing_structure_pipeline(dataset)
                else:
                    db_feature = featurizing_structure_pipeline(dataset)

            # composition pipeline
            elif mode == "composition":
                tqdm.pandas(desc="Processing formulae")
                dataset["formula"] = dataset["composition"].apply(lambda x: x.reduced_formula)
                if self.target_property in dataset.columns:
                    dataset[self.target_property] = dataset[self.target_property].astype(float)

                # Run the featurization pipeline
                if isinstance(databases[0], GNoMEDatabase):
                    db_feature = featurizing_composition_pipeline(dataset)
                else:
                    db_feature = featurizing_composition_pipeline(dataset)

            else:
                logger.error("Invalid featurization mode. Must be one of ['structure', 'composition'].")
                raise ValueError("mode must be one of ['structure', 'composition'].")

            # Retain only numerical features
            db_feature = db_feature.select_dtypes(exclude=["object", "bool"])
            if self.target_property in dataset.columns:
                db_feature[self.target_property] = dataset.set_index("formula")[self.target_property]
            db_feature.dropna(inplace=True)  # Remove NaN rows

            # Adjust max_dof if needed
            if max_dof is None:
                n_items, n_features = db_feature.shape
                self.max_dof = min(n_items // 10, n_features - 1)

        return db_feature

    def _build_regressor(self, n_jobs: int = 1, max_dof: int = None, scoring: str = "neg_root_mean_squared_error"):
        """
        Builds a Gradient Boosting Regressor pipeline with feature selection.

        This method constructs a regression pipeline that includes:
        1. Standard scaling of features.
        2. Feature selection using Recursive Feature Elimination (RFE) with a Gradient Boosting Regressor
           as the estimator.
        3. A Gradient Boosting Regressor as the final model for regression.

        It performs a grid search to tune hyperparameters for the number of estimators and the number of
        features to select.
        The search uses K-Folds cross-validation and evaluates performance using the RMSE score.

        Args:
            n_jobs (int, optional): The number of CPU cores to use for parallel processing during the grid search.
                Default is 1.
            scoring (str, optional): The strategy to evaluate the performance of the cross-validated model on
                the validation set.

        Returns:
            (GridSearchCV): A GridSearchCV object that encapsulates the regressor pipeline and hyperparameter
                tuning process.

        Notes:
            - The `n_estimators` in the regressor will be searched over the values [50, 100, 250, 500].
            - The number of features to select for RFE is determined by the `max_dof` argument,
              with values [int(max_dof * 0.5), int(max_dof * 1)].
            - K-Folds cross-validation is used with 4 splits, shuffling enabled, and a fixed random seed (0).
            - R2 score is used for model evaluation.
        """
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("feature_selector", RFE(estimator=GradientBoostingRegressor(), step=25)),
                ("regressor", GradientBoostingRegressor()),
            ],
        )
        param_grid = {
            "regressor__n_estimators": [50, 100, 250, 500],
            "feature_selector__n_features_to_select": [int(max_dof * 0.5), int(max_dof * 1)],
        }

        kfold = KFold(n_splits=4, shuffle=True, random_state=0)
        search = GridSearchCV(pipe, param_grid, n_jobs=n_jobs, verbose=2, cv=kfold, scoring=scoring)

        return search

    def compile(self, n_jobs: int = 1, max_dof: int = None, scoring: str = "neg_root_mean_squared_error"):  # noqa:A003
        """
        Initializes the regressor pipeline and sets up the hyperparameter search.

        This method calls the `_build_regressor` method to construct a Gradient Boosting Regressor pipeline,
        including feature scaling, feature selection via Recursive Feature Elimination (RFE), and regression.
        It then stores the resulting `GridSearchCV` object in the `self.search` attribute.

        Args:
            n_jobs (int, optional): The number of CPU cores to use for parallel processing during the grid search.
                Default is 1.
            max_dof (int, optional): The maximum degrees of freedom for feature selection. If provided,
                it will control the number of features to select.

        Returns:
            None

        Notes:
            This method does not return any value but sets the `self.search` attribute with the initialized
            `GridSearchCV` object, which contains the regressor pipeline and hyperparameter tuning setup.
        """
        if max_dof is None:
            max_dof = self.max_dof

        logger.info("Build regressors")
        self.search = self._build_regressor(n_jobs=n_jobs, max_dof=max_dof, scoring=scoring)

    def fit(self, df: pd.DataFrame):
        """
        Train and save multiple models using a `GridSearchCV` regressor.

        This method iterates through the specified number of committers (`n_committers`) and performs the following:
        1. Trains a model for each committer using the `GridSearchCV` pipeline (`self.search`) with the given
           data (`df`).
        2. Saves each trained model as a `.pkl` file in the specified `self.models_dir`.

        Args:
            df (pd.DataFrame): The input dataset. The last column is assumed to be the target variable,
                and all other columns are used as features.

        Returns:
            None

        Notes:
            - The models are saved as `.pkl` files in the `self.models_dir` directory, with filenames
            following the pattern `{self._model_spec}.rep{i}.pkl`, where `i` is the index of the committer.
            - The `GridSearchCV` pipeline, defined in `self.search`, is used for training.
        """
        models = {}
        for i in tqdm(range(self.n_committers), desc="training models"):
            models[f"model_{i}"] = self.search.fit(df.iloc[:, :-1], df.iloc[:, -1])

        for i in tqdm(range(self.n_committers), desc="saving"):
            model_path = self.models_dir / (self._model_spec + f".rep{i}.pkl")
            model_ = models[f"model_{i}"]
            with open(model_path, "wb") as file_:
                pickle.dump(model_, file_)

    def evaluate(self, df: pd.DataFrame, return_df: bool = False):
        """
        Evaluate the performance of multiple models on a given dataset.

        This method evaluates each model's predictions on the provided dataset (`df`) and returns the predictions
        either as a DataFrame with true values and model predictions or as a dictionary of model predictions.

        Args:
            df (pd.DataFrame): The dataset containing the features and the target property (last column).
                The model predictions are based on all columns except the last one (target property).
            return_df (bool, optional): If True, returns a DataFrame with true values and predictions from each model.
                If False, returns a dictionary with model predictions. Defaults to `False`.

        Returns:
            (pd.DataFrame): If `return_df=True`, returns a pandas DataFrame where each column
                corresponds to predictions and loss for each model.
                The columns include:
                    - `true_value`: Ground truth values.
                    - `model_i_prediction`: Predictions from model `i`.

            (dict[str, pd.DataFrame]): If `return_df=False`, returns a dictionary where each key is
                a model identifier (e.g., `model_0`, `model_1`, ...) and the value is a
                DataFrame containing the following columns:
                    - `true_value`: Ground truth values.
                    - `prediction`: Predictions from the model.

        Notes:
            - The target property in `df` must match `self.target_property`.
            - The method assumes the target property is in the last column of the input `df` and features are in all
              other columns.
        """
        if return_df:
            predictions = pd.DataFrame(columns=["true"])
            predictions["true"] = df[self.target_property]

            for i in tqdm(range(self.n_committers), desc="models"):
                predictions[f"model_{i}"] = np.empty((len(df), 1)).tolist()
                predictions[f"model_{i}"] = self.models[f"model_{i}"].predict(df.iloc[:, :-1])[:]

        else:
            predictions = {}
            for i in tqdm(range(self.n_committers), desc="models"):
                predictions[f"model_{i}"] = pd.DataFrame()
                predictions[f"model_{i}"]["true_value"] = df[self.target_property]
                predictions[f"model_{i}"]["prediction"] = self.models[f"model_{i}"].predict(df.iloc[:, :-1])[:]

        return predictions

    def plot_parity(
        self,
        predictions_dict: dict[str, pd.DataFrame],
        include_ensemble: bool = True,
        fig_settings: dict[str, Any] = dict(figsize=(5, 3), dpi=100),
    ):
        """
        Plot a parity plot for model predictions and their comparison with true values.

        This method generates a scatter plot where the x-axis represents the true values,
        and the y-axis represents the predicted values from one or more models. It also
        includes a reference line (1:1 line) and error histograms as insets to visualize
        the prediction error distribution. Additionally, it calculates and annotates the R²
        value for each model's predictions and optionally for the ensemble average of all models.

        Args:
            predictions_dict (dict): A dictionary where keys are model names (e.g., 'model_1', 'model_2')
                and values are pandas DataFrames containing the `true_value` and `prediction` columns.
            include_ensemble (bool): If `True`, an ensemble prediction (mean of all model predictions)
                is included in the plot. Default is `True`.
            fig_settings (dict): A dictionary containing matplotlib figure size arguments (see
            [`matplotlib.pyplot.figure`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html))
        """
        all_predictions = []
        fig, ax = plt.subplots(**fig_settings)

        colors = plt.cm.tab10.colors  # Get distinct colors for different models

        for i, (model, df) in enumerate(predictions_dict.items()):
            y_true = df["true_value"]
            y_predictions = df["prediction"]

            if include_ensemble:
                all_predictions.append(y_predictions)

            error = np.abs(y_predictions - y_true)
            r2 = r2_score(y_true, y_predictions)

            # Scatter plot with unique color
            ax.scatter(
                y_true,
                y_predictions,
                s=6,
                alpha=0.6,
                color=colors[i % len(colors)],
                label=model,
                zorder=1,
            )

            # Reference line (1:1 line)
            ax.axline((np.mean(y_true), np.mean(y_true)), slope=1, lw=0.85, ls="--", color="k", zorder=2)

            # Add inset histogram
            if i == 0:  # Create inset only once
                axin = ax.inset_axes([0.65, 0.17, 0.3, 0.3])
            axin.hist(error, bins=int(np.sqrt(len(error))), alpha=0.6, color=colors[i % len(colors)])
            axin.hist(error, bins=int(np.sqrt(len(error))), histtype="step", lw=1, color="black")

            # Annotate R² values dynamically
            ax.annotate(
                f"$R^2={r2:1.2f}$",
                xy=(0.05, 0.96 - (i * 0.07)),  # Adjust position dynamically
                xycoords="axes fraction",
                va="top",
                ha="left",
                fontsize=8,
                color=colors[i % len(colors)],
            )

        # Add ensemble prediction plot (if required)
        if include_ensemble:
            ensemble_predictions = np.mean(all_predictions, axis=0)
            ensemble_r2 = r2_score(y_true, ensemble_predictions)

            # Scatter plot for ensemble prediction
            ax.scatter(
                y_true,
                ensemble_predictions,
                s=6,
                alpha=0.6,
                color=colors[len(predictions_dict) % len(colors)],
                label="Ensemble",
                zorder=3,
            )

            # Annotate R² value for ensemble
            ax.annotate(
                f"Ensemble $R^2={ensemble_r2:1.2f}$",
                xy=(0.05, 0.96 - (len(predictions_dict) * 0.07)),  # Adjusted position for ensemble
                xycoords="axes fraction",
                va="top",
                ha="left",
                fontsize=8,
                color=colors[len(predictions_dict) % len(colors)],
            )

        # Set labels
        ax.set_xlabel("True value")
        ax.set_ylabel("Predicted value")

        # Set axis limits to ensure a square parity plot
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        min_ = min(min(xlim), min(ylim))
        max_ = max(max(xlim), max(ylim))
        ax.set_xlim(min_, max_)
        ax.set_ylim(min_, max_)

        # Move legend outside the plot
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=8, borderaxespad=0.0)

        # Add title
        fig.suptitle("Parity Plot", fontsize=10)

        # Save figures
        fig.savefig(self.figures_dir / (self._model_spec + "_parity.png"), dpi=330, bbox_inches="tight")
        fig.savefig(self.figures_dir / (self._model_spec + "_parity.pdf"), dpi=330, bbox_inches="tight")

        # Show plot
        plt.show()

    def _evaluate_unknown(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate and predict the target property using a committee of regressors.

        This method takes a DataFrame of input features, applies a set of pre-trained regressors
        (stored in `self.models`), and generates probability predictions for each regressor in the committee.
        It then computes the mean and standard deviation of the regressor predictions to provide an
        overall prediction and uncertainty.

        Args:
            df (pd.DataFrame): A DataFrame where each row represents a sample, and the columns represent features.
                The last column is assumed to be the target property, which is not used in prediction.

        Returns:
            (pd.DataFrame): A DataFrame containing:
                - The predicted probability from each regressor in the committee (labeled as `regressor_{i}`).
                - The mean of all regressor predictions (`regressor_mean`).
                - The standard deviation of the regressor predictions (`regressor_std`).

        Notes:
            - The input `df` should have one or more features for prediction, and the last column is ignored
              during prediction.
            - The `regressor_mean` column represents the average of all regressors' probability predictions.
            - The `regressor_std` column gives a measure of the variance (uncertainty) of the predictions across
              the regressors.
        """
        predictions = pd.DataFrame(index=df.index)

        for i in tqdm(range(self.n_committers), desc="models"):
            predictions[f"regressor_{i}"] = self.models[f"model_{i}"].predict(df)[:]
        predictions["regressor_mean"] = predictions[[f"regressor_{i}" for i in range(self.n_committers)]].mean(axis=1)
        predictions["regressor_std"] = predictions[[f"regressor_{i}" for i in range(self.n_committers)]].std(axis=1)
        return predictions

    def predict(
        self,
        db: GNoMEDatabase,
        stage: str = "processed",
        confidence_threshold: float = 0.5,
        featurizing_mode: str = "structure",
        save_final: bool = True,
    ) -> pd.DataFrame:
        """
        Predicts the target property for candidate specialized materials using regressor models,
        after filtering materials based on regressor committee confidence.

        Args:
            db (GNoMEDatabase): The database containing the materials and their properties.
            stage (str): The processing stage ("raw", "processed", "final"). Defaults to `processed`.
            confidence_threshold (float, optional): The minimum regressor committee confidence
                required to keep a material for prediction.
                Defaults to `0.5`.
            featurizing_mode (str, optional): Type of featurization to apply. Options:
                - "structure": Parses `.cif` files and applies structure-based features.
                - "composition": Uses only the chemical composition for features.
                Defaults to "structure".
            save_final (bool, optional): Whether to save the final database with predictions.
                Defaults to `True`.

        Returns:
            (pd.DataFrame): A DataFrame containing the predictions, along with the true values and
                regressor committee confidence scores for the screened materials.

        Notes:
            - The method filters the materials based on the regressor confidence, then uses the
            regressor models to predict the target property for the remaining materials.
            - If `save_final` is set to True, the predictions are saved to the database in the
            `final` stage.
        """
        logger.info(f"Discarding materials with regressor committee confidence threshold < {confidence_threshold}.")
        logger.info("Featurizing the database.")
        df_feat = self.featurize_db(db, mode=featurizing_mode, confidence_threshold=confidence_threshold)
        logger.info("Predicting the target property for candidate specialized materials.")
        predictions = self._evaluate_unknown(df_feat)
        df = db.get_database(stage)
        screened = df[df["classifier_mean"] > confidence_threshold]
        predictions = pd.concat([screened.reset_index(), predictions.reset_index()], axis=1)

        if save_final:
            logger.info("Saving the final database.")
            db.databases["final"] = predictions.copy()
            db.save_database("final")

        return predictions
