import os
from pathlib import Path
import pickle
import warnings

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Structure
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from energy_gnome.config import DEFAULT_GBDT_SETTINGS, FIGURES_DIR, MODELS_DIR
from energy_gnome.dataset.base_dataset import BaseDatabase
from energy_gnome.dataset.gnome import GNoMEDatabase
from energy_gnome.models.abc_model import BaseClassifier
from energy_gnome.utils.readers import load_yaml, save_yaml

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
        Initialize the GBDTClassifier with directories for storing models and figures.

        This class extends `BaseClassifier` to implement a gradient boosted decision tree (GBDT)
        for classification tasks. It sets up the necessary directory structure and configurations
        for training models.

        Args:
            model_name (str): Name of the model, used to create subdirectories.
            models_dir (Path | str, optional): Directory for storing trained model weights.
                                            Defaults to MODELS_DIR from config.
            figures_dir (Path | str, optional): Directory for saving figures and visualizations.
                                                Defaults to FIGURES_DIR from config.

        Attributes:
            _model_spec (str): Specification string used for model identification.
        """
        self._model_spec = "model.gbdt_classifier"
        super().__init__(model_name=model_name, models_dir=models_dir, figures_dir=figures_dir)

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
        subset: str | None = None,
        max_dof: int = None,
        mute_warnings: bool = True,
    ) -> pd.DataFrame:
        """
        Load and featurize the specified databases efficiently.

        This method processes the given list of databases and applies a featurization pipeline to each dataset.
        It handles the loading of raw data from databases, extracts relevant features such as composition and
        structure, and applies transformations to create numerical features for model training.

        Args:
            databases (list[BaseDatabase]): A list of databases (or a single database) from which to
                                            load and featurize data.
            subset (str, optional): The subset of data to load from each database (default is None).
            max_dof (int, optional): Maximum degrees of freedom for the feature space. If not provided,
                                        it is automatically calculated.
            mute_warnings (bool, optional): Whether to suppress warnings during featurization (default is True).

        Returns:
            (pd.DataFrame): A DataFrame containing the featurized data, including numerical features
                            and a column for `is_specialized`.

        Warnings:
            If `mute_warnings` is set to False, third-party warnings related to the featurization
            process will be displayed.

        Notes:
            - If the database contains a `Reduced Formula` or `formula_pretty` column, compositions are
              parsed accordingly.
            - The resulting DataFrame retains only numerical features, removes NaN rows, and includes an
              `is_specialized` column.
            - The method supports batch processing using `tqdm` for progress visualization during featurization.
        """

        if not isinstance(databases, list):
            databases = [databases]

        if mute_warnings:
            logger.warning("Third-party warnings disabled. Set 'mute_warnings=False' to enable them.")
            self.max_dof = max_dof

        if isinstance(databases[0], GNoMEDatabase):
            dataset = databases[0].get_database("raw")
        else:
            # Load all database subsets efficiently
            dataset = pd.concat([db.load_classifier_data(subset) for db in databases], ignore_index=True)

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
            dataset["structure"] = dataset["cif_path"].progress_apply(lambda x: Structure.from_file(x))
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

    def _build_classifier(self, n_jobs: int = 1, max_dof: int = None):
        """
        Builds a Gradient Boosting Classifier pipeline with feature selection.

        This method constructs a classification pipeline that includes:
        1. Standard scaling of features.
        2. Feature selection using Recursive Feature Elimination (RFE) with a Gradient Boosting Classifier
           as the estimator.
        3. A Gradient Boosting Classifier as the final model for classification.

        It performs a grid search to tune hyperparameters for the number of estimators and the number of
        features to select.
        The search uses Stratified K-Folds cross-validation and evaluates performance using the ROC-AUC score.

        Args:
            n_jobs (int, optional): The number of CPU cores to use for parallel processing during the grid search.
                                    Default is 1.
            max_dof (int, optional): The maximum degrees of freedom for feature selection. If provided,
                                     it will control the number of features to select.

        Returns:
            (GridSearchCV): A GridSearchCV object that encapsulates the classifier pipeline and hyperparameter
                            tuning process.

        Notes:
            - The `n_estimators` in the classifier will be searched over the values [50, 100, 250, 500].
            - The number of features to select for RFE is determined by the `max_dof` argument,
              with values [int(max_dof * 0.5), int(max_dof * 1)].
            - Stratified K-Folds cross-validation is used with 4 splits, shuffling enabled, and a fixed random seed (0).
            - ROC-AUC score is used for model evaluation.
        """
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
        search = GridSearchCV(pipe, param_grid, n_jobs=n_jobs, verbose=2, cv=stratified_kfold, scoring="roc_auc")

        return search

    def compile_(  # same here, _ to mute the pre-commit
        self,
        n_jobs: int = 1,
    ):
        """
        Initializes the classifier pipeline and sets up the hyperparameter search.

        This method calls the `_build_classifier` method to construct a Gradient Boosting Classifier pipeline,
        including feature scaling, feature selection via Recursive Feature Elimination (RFE), and classification.
        It then stores the resulting `GridSearchCV` object in the `self.search` attribute.

        Args:
            n_jobs (int, optional): The number of CPU cores to use for parallel processing during the grid search.
                                    Default is 1.

        Returns:
            None

        Notes:
            This method does not return any value but sets the `self.search` attribute with the initialized
            `GridSearchCV` object, which contains the classifier pipeline and hyperparameter tuning setup.
        """

        logger.info("Build classifiers")
        self.search = self._build_classifier(n_jobs=n_jobs, max_dof=self.max_dof)

    def fit(self, df: pd.DataFrame):
        """
        Train and save multiple models using a `GridSearchCV` classifier.

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
            - Each model's predictions are generated using the `predict_proba` method, which is expected to
              return probabilities.
            - The method assumes the target property is in the last column of the input `df` and features are in all
              other columns.
        """
        if return_df:
            predictions = pd.DataFrame(columns=["true"])
            predictions["true"] = df[self.target_property]

            for i in tqdm(range(self.n_committers), desc="models"):
                predictions[f"model_{i}"] = np.empty((len(df), 1)).tolist()
                predictions[f"model_{i}"] = self.models[f"model_{i}"].predict_proba(df.iloc[:, :-1])[:, 1]

        else:
            predictions = {}
            for i in tqdm(range(self.n_committers), desc="models"):
                predictions[f"model_{i}"] = pd.DataFrame()
                predictions[f"model_{i}"]["true_value"] = df[self.target_property]
                predictions[f"model_{i}"]["prediction"] = self.models[f"model_{i}"].predict_proba(df.iloc[:, :-1])[:, 1]

        return predictions

    def plot_performance(self, predictions_dict: dict[str, pd.DataFrame], include_ensemble: bool = True):
        """
        Plot model performance evaluation curves: ROC, Precision, and Recall.

        This method generates a multi-panel plot that visualizes the performance of different models on
        classification tasks. It includes:
        - ROC curve with AUC (Area Under the Curve)
        - Precision-Recall curve
        - Recall-Threshold curve

        The method also supports an optional ensemble model performance evaluation by averaging
        individual model predictions.

        Args:
            predictions_dict (dict[str, pd.DataFrame]): A dictionary where keys are model names and values are
                                                        DataFrames containing the `true_value` and `prediction` columns.
                                                        Each model's predictions will be plotted.
            include_ensemble (bool, optional): If `True`, the ensemble model performance will also be plotted, which is
                                            based on averaging the predictions of all models. Defaults to `True`.

        Returns:
            None: The method generates and saves the performance plots as PNG and PDF files.

        Notes:
            - The method assumes that the `predictions_dict` contains the model predictions (in the `prediction` column)
            and the true labels (in the `true_value` column).
            - The ROC curve is evaluated using the `roc_curve` function, while the Precision and Recall curves are
            generated using `precision_recall_curve`.
            - The final figure is saved in both PNG and PDF formats in the directory defined by `self.figures_dir`.
        """

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

    def _evaluate_unknown(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate and predict the target property using a committee of classifiers.

        This method takes a DataFrame of input features, applies a set of pre-trained classifiers
        (stored in `self.models`), and generates probability predictions for each classifier in the committee.
        It then computes the mean and standard deviation of the classifier predictions to provide an
        overall prediction and uncertainty.

        Args:
            df (pd.DataFrame): A DataFrame where each row represents a sample, and the columns represent features.
                                The last column is assumed to be the target property, which is not used in prediction.

        Returns:
            (pd.DataFrame): A DataFrame containing:
                - The predicted probability from each classifier in the committee (labeled as `classifier_{i}`).
                - The mean of all classifier predictions (`classifier_mean`).
                - The standard deviation of the classifier predictions (`classifier_std`).

        Notes:
            - The input `df` should have one or more features for prediction, and the last column is ignored
              during prediction.
            - The method assumes that the classifiers in `self.models` are capable of generating probability predictions
            using the `predict_proba` method, and that the classifiers are indexed from `0` to `n_committers-1`.
            - The `classifier_mean` column represents the average of all classifiers' probability predictions.
            - The `classifier_std` column gives a measure of the variance (uncertainty) of the predictions across
              the classifiers.
        """
        predictions = pd.DataFrame(index=df.index)

        for i in tqdm(range(self.n_committers), desc="models"):
            predictions[f"classifier_{i}"] = self.models[f"model_{i}"].predict_proba(df.iloc[:, :-1])[:, 1]
        predictions["classifier_mean"] = predictions[[f"classifier_{i}" for i in range(self.n_committers)]].mean(axis=1)
        predictions["classifier_std"] = predictions[[f"classifier_{i}" for i in range(self.n_committers)]].std(axis=1)
        return predictions

    def screen(self, db: GNoMEDatabase, save_processed: bool = True) -> pd.DataFrame:
        """
        Screen the database for specialized materials using classifier predictions.

        This method performs the following steps:
        1. Featurizes the database using `featurize_db`.
        2. Evaluates the featurized data using a committee of classifiers to generate predictions.
        3. Combines the predictions with the original database and removes rows with missing values
           or unqualified materials.
        4. Optionally saves the processed (screened) database for future use.

        Args:
            db (GNoMEDatabase): A `GNoMEDatabase` object containing the raw data to be screened.
            save_processed (bool, optional): Whether to save the screened data to the database. Defaults to `True`.

        Returns:
            (pd.DataFrame): A DataFrame containing the original data combined with classifier predictions,
                        excluding materials that have missing or unqualified values for screening.

        Notes:
            - The method assumes that `featurize_db` and `_evaluate_unknown` methods are defined and function correctly.
            - The `classifier_mean` column in the returned DataFrame reflects the mean classifier prediction, which is
              used to screen specialized materials.
            - The `is_specialized` column is dropped from the screened DataFrame.
        """
        logger.info("Featurizing the database...")
        df_class = self.featurize_db(db)
        logger.info("Screening the database for specialized materials.")
        predictions = self._evaluate_unknown(df_class)
        gnome_df = db.get_database("raw")
        gnome_screened = pd.concat([gnome_df, predictions.reset_index(drop=True)], axis=1)
        gnome_screened.drop(columns=["is_specialized"], inplace=True)
        gnome_screened = gnome_screened[gnome_screened["classifier_mean"].notna()]

        if save_processed:
            logger.info("Saving the screened database.")
            db.databases["processed"] = gnome_screened.copy()
            db.save_database("processed")

        return gnome_screened
