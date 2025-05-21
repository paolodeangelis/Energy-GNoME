import asyncio
import json
import os
from pathlib import Path
from typing import Any

from ase.io import read
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from energy_gnome.config import (
    DEFAULT_E3NN_SETTINGS,
    DEFAULT_OPTIM_SETTINGS,
    DEFAULT_TRAINING_SETTINGS,
    FIGURES_DIR,
    MODELS_DIR,
)
from energy_gnome.dataset.base_dataset import BaseDatabase
from energy_gnome.dataset.gnome import GNoMEDatabase
from energy_gnome.models.abc_model import BaseRegressor
from energy_gnome.models.e3nn.utils import (
    PeriodicNetwork,
    build_data,
    get_encoding,
    get_neighbors,
    train_regressor,
)
from energy_gnome.utils.readers import load_json, load_yaml, save_yaml, to_unix

DEFAULT_DTYPE = torch.float64
tqdm.pandas()
torch.set_default_dtype(DEFAULT_DTYPE)
mp.set_start_method("spawn", force=True)


def _train_model(
    i: int,
    device: str,
    model_spec: str,
    model: torch.nn.Module,
    models_dir: Path,
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    weight_decay: float,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scheduler_settings: dict[str, Any],
    loss_function: torch.nn.Module,
    n_epochs: int,
    dataloader_train: DataLoader,
    dataloader_valid: DataLoader,
) -> None:
    """
    Train a single model instance.

    This function initializes a model, sets it up with an optimizer and scheduler,
    and trains it using the provided data.

    Args:
        i (int): Model index for logging and saving purposes.
        device (str): The computing device (e.g., "cuda:0" or "cpu").
        model_spec (str): Specification identifier for the model.
        model (torch.nn.Module): The PyTorch model to be trained.
        models_dir (Path): Directory where model checkpoints are stored.
        optimizer (Type[torch.optim.Optimizer]): Optimizer class (e.g., AdamW).
        learning_rate (float): Learning rate for training.
        weight_decay (float): Weight decay (L2 regularization).
        scheduler (Type[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler class.
        scheduler_settings (dict[str, Any]): Configuration dictionary for the scheduler.
        loss_function (Type[torch.nn.Module]): Loss function class (e.g., L1Loss).
        n_epochs (int): Number of training epochs.
        dataloader_train (DataLoader): Dataloader for training data.
        dataloader_valid (DataLoader): Dataloader for validation data.

    Returns:
        None
    """

    torch.cuda.set_device(device)
    model_path = models_dir / (model_spec + f".rep{i}")
    # model_ = models[f"model_{i}"]
    model_ = model
    logger.debug(f"Running model_{i} on device={device}, and stored in {model_path}")
    model_.pool = True
    model_.to(device)

    # Compile
    opt = optimizer(model_.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler_instance = scheduler(opt, **scheduler_settings)
    loss_fn = loss_function()

    train_regressor(
        model_,
        opt,
        dataloader_train,
        dataloader_valid,
        loss_fn,
        model_path,
        max_epoch=n_epochs,
        scheduler=scheduler_instance,
        only_best=True,
        device=device,
    )


class E3NNRegressor(BaseRegressor):
    def __init__(
        self,
        model_name: str,
        target_property: str,
        models_dir: Path | str = MODELS_DIR,
        figures_dir: Path | str = FIGURES_DIR,
    ):
        """
        Initialize the E3NNRegressor with directories for storing models and figures.

        This class extends `BaseRegressor` to implement an equivariant neural network (E3NN)
        for regression tasks. It sets up the necessary directory structure and configurations
        for training models.

        Args:
            model_name (str): Name of the model, used to create subdirectories.
            target_property (str): The target property the model is trained to predict.
            models_dir (Path | str, optional): Directory for storing trained model weights.
                                            Defaults to `MODELS_DIR` from config.
            figures_dir (Path | str, optional): Directory for saving figures and visualizations.
                                                Defaults to `FIGURES_DIR` from config.

        Attributes:
            _model_spec (str): Specification string used for model identification.
            l_max (int): Maximum order of the spherical harmonics used in the E3NN model (default is 2).
            r_max (int): Cutoff radius used in the E3NN model (default is 4).
            conv_layers (int): Number of nonlinearities (number of convolutions = layers + 1, default is 2).
        """

        self._model_spec = "model.e3nn_regressor." + target_property
        super().__init__(
            model_name=model_name,
            target_property=target_property,
            models_dir=models_dir,
            figures_dir=figures_dir,
        )
        self.l_max: int = 2
        self.r_max: int = 4
        self.conv_layers: int = 2

    def _find_model_states(self):
        models_states = []
        if any(self.models_dir.iterdir()):
            models_states = [f_ for f_ in self.models_dir.iterdir() if f_.match("*.torch")]
        return models_states

    def load_trained_models(self, state: str = "state_best"):
        """
        Load trained models from the model directory.

        This method searches for trained models by:
        1. Loading model settings from `.yaml` files matching `_model_spec`.
        2. Initializing models based on corresponding `.json` configuration files.
        3. Loading the model weights from `.torch` files.
        4. Storing the loaded models in `self.models`.

        Args:
            state (str, optional): The key used to extract model weights from the saved
                                state dictionary (e.g., `"state_best"`). Defaults to `"state_best"`.

        Returns:
            list[str]: A list of `.torch` model filenames that were found in the directory.
        """
        for yaml_path in self.models_dir.glob(f"*{self._model_spec}.yaml"):
            self.set_model_settings(yaml_file=yaml_path)

        i = 0
        loaded_models = []
        for model_path in self.models_dir.glob("*.torch"):
            if self._model_spec in model_path.name:
                model_setting_path = model_path.with_suffix(".json")
                if model_setting_path.exists():
                    try:
                        logger.info(f"Loading model with setting in {model_setting_path}")
                        logger.info(f"And weights in {model_path}")
                        model_setting = load_json(model_setting_path)

                        model = PeriodicNetwork(**model_setting)
                        model.load_state_dict(
                            torch.load(model_path, map_location=self.device, weights_only=True)[
                                state
                            ]
                        )
                        model.pool = True
                        self.models[f"model_{i}"] = model
                        loaded_models.append(model_path.name)
                        i += 1
                    except Exception as e:
                        logger.error(f"Error loading model {model_path.name}: {e}")
                else:
                    logger.warning(f"Missing JSON settings for model weights in {model_path.name}")

        return loaded_models

    def _load_model_setting(self, yaml_path):
        """
        Load model settings from a YAML file and assign corresponding attributes.

        This method loads settings from the specified YAML file and sets model attributes
        based on the values in `DEFAULT_E3NN_SETTINGS`. If an attribute is missing, a KeyError
        will be raised.

        Args:
            yaml_path (Path): Path to the YAML file containing the model settings.
        """
        try:
            settings = load_yaml(yaml_path)
            for att, _ in DEFAULT_E3NN_SETTINGS.items():
                setattr(self, att, settings[att])

            # Ensure that the device is set correctly
            if "cuda" in self.device and not torch.cuda.is_available():
                logger.warning(f"Models trained on {self.device} but only the CPU is available.")
                self.device = "cpu"
        except Exception as e:
            logger.error(f"Error loading model settings from {yaml_path}: {e}")
            raise

    def _save_model_settings(self):
        """
        Save the current model settings to a YAML file.

        This method saves the current values of the model's attributes (based on `DEFAULT_E3NN_SETTINGS`)
        to a YAML file in the models directory.

        The file is named based on the model specification (`self._model_spec`) with a `.yaml` extension.

        Raises:
            IOError: If the saving process fails.
        """
        try:
            settings_path = self.models_dir / (self._model_spec + ".yaml")
            settings = {att: getattr(self, att) for att, _ in DEFAULT_E3NN_SETTINGS.items()}

            logger.info(f"Saving model settings to {settings_path}")
            save_yaml(settings, settings_path)
        except Exception as e:
            logger.error(f"Error saving model settings to {settings_path}: {e}")
            raise

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

        # Accessing model settings (DEFAULT or provided in kargs)
        for att, defvalue in DEFAULT_E3NN_SETTINGS.items():
            if att in kargs:
                # If a setting is provided via kargs, use it
                setattr(self, att, kargs[att])
            else:
                try:
                    # Check if the attribute already exists and is not None
                    att_exist = getattr(self, att)
                    # If the attribute exists, we verify it's not None (NaN check)
                    att_exist = att_exist == att_exist
                except AttributeError:
                    # If the attribute does not exist, it will be set to the default value
                    att_exist = False

                if not att_exist:
                    # If the attribute doesn't exist or is None, use the default value
                    setattr(self, att, defvalue)
                    logger.warning(f"Using default value {defvalue} for {att} setting")

        # If yaml_file was not provided or is in a different directory, save settings
        if yaml_file is None or os.path.dirname(str(yaml_file)) != str(self.models_dir):
            self._save_model_settings()

    def set_training_settings(self, n_epochs: int):
        """
        Set the number of epochs for training.

        This method sets the number of epochs for the model's training process.
        It is assumed that the training process will be carried out for the specified
        number of epochs.

        Args:
            n_epochs (int): The number of epochs for training.
                            It should be a positive integer.
        """
        self.n_epochs = n_epochs

    def set_optimizer_settings(self, lr: float, wd: float):
        """
        Set the optimizer settings, including learning rate and weight decay.

        This method sets the learning rate and weight decay for the optimizer, which
        will be used in the training process.

        Args:
            lr (float): The learning rate for the optimizer. It should be a positive float.
            wd (float): The weight decay (regularization) parameter for the optimizer.
                        It should be a non-negative float.
        """
        self.learning_rate = lr
        self.weight_decay = wd

    def featurize_db(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Featurize the given dataset by processing the CIF file paths and extracting
        structural and chemical information.

        This method reads the CIF files specified in the input dataset, extracts chemical
        information (such as formulae and species), and then generates featurized data suitable
        for model training. It also preserves the target property if it exists in the dataset.

        Args:
            dataset (pd.DataFrame): Input dataset containing the CIF file paths and optionally
                                    the target property.

        Returns:
            (pd.DataFrame): The featurized dataset, including chemical information and features
                        for model training. The dataset includes columns for 'structure', 'species',
                        'formula', and 'data' (featurized data).
        """
        if dataset.empty:
            return pd.DataFrame()

        # Initialize tqdm for apply functions
        tqdm.pandas()

        # Create a copy to avoid modifying the original dataset
        # database_nn = dataset.copy()
        database_nn = pd.DataFrame(
            {
                "structure": pd.Series(dtype="object"),
                "species": pd.Series(dtype="object"),
            }
        )

        # Read all structures in parallel with progress tracking
        database_nn["structure"] = dataset["cif_path"]
        database_nn["structure"] = database_nn["structure"].progress_apply(
            lambda path: read(Path(to_unix(path)))
        )

        # Extract formula and species info
        database_nn["formula"] = database_nn["structure"].progress_apply(
            lambda s: s.get_chemical_formula()
        )
        database_nn["species"] = database_nn["structure"].progress_apply(
            lambda s: list(set(s.get_chemical_symbols()))
        )

        # Preserve target property
        if self.target_property in dataset.columns:
            database_nn[self.target_property] = dataset[self.target_property]

        # Get encoding for featurization
        type_encoding, type_onehot, atomicmass_onehot = get_encoding()
        r_max = self.r_max  # cutoff radius

        # Apply featurization with progress bar
        database_nn["data"] = database_nn.progress_apply(
            lambda x: build_data(
                x,
                type_encoding,
                type_onehot,
                atomicmass_onehot,
                self.target_property if self.target_property in database_nn.columns else None,
                r_max,
                dtype=DEFAULT_DTYPE,
            ),
            axis=1,
        )

        return database_nn

    def create_dataloader(
        self,
        databases: BaseDatabase | list[BaseDatabase],
        subset: str | None = None,
        shuffle: bool = False,
        confidence_threshold: float = 0.5,
    ):
        """
        Format and return a PyTorch DataLoader for training, validation, or testing.

        This method prepares the dataloaders for PyTorch training by featurizing the input datasets
        and handling multiple types of databases. It ensures that shuffling is only applied to the
        training subset, and it filters out low-confidence samples from GNoMEDatabase.

        Args:
            databases (BaseDatabase | list[BaseDatabase]): A single instance or a list of `BaseDatabase` objects
                                                        containing processed data. If multiple databases are provided,
                                                        they will be concatenated.
            subset (str, optional): The specific data subset to use (`training`, `validation`, or `testing`).
                                    If `None`, all data will be used. Defaults to `None`.
            shuffle (bool, optional): Whether to shuffle the data. This is only applicable for the training set.
                                    Defaults to `False`.
            confidence_threshold (float, optional): The threshold for filtering out low-confidence entries in the
                                                    `GNoMEDatabase`. Defaults to `0.5`.

        Raises:
            ValueError: If shuffling is set to `True` for the validation or testing subset.

        Returns:
            (tuple): A tuple containing:
                - dataloader_db (DataLoader): The PyTorch DataLoader object containing the processed data.
                - mean_neighbors (float): The mean number of neighbors (calculated using the featurized data).
        """
        if subset != "training" and shuffle is True:
            logger.error("Shuffling is not supported for the validation and testing set!")
            raise ValueError("Shuffling is not supported for the validation and testing set!")

        if not isinstance(databases, list):
            databases = [databases]

        if isinstance(databases[0], GNoMEDatabase):
            df = databases[0].get_database("processed")
            df = df[df["classifier_mean"] > confidence_threshold]
        else:
            db_list = []
            for db in databases:
                db_list.append(db.load_regressor_data(subset))

            if len(db_list) > 1:
                df = pd.concat(db_list, ignore_index=True)
            else:
                df = db_list[0]
                df.reset_index(drop=True, inplace=True)

        featurized_db = self.featurize_db(df)

        data_values = featurized_db["data"].values
        dataloader_db = DataLoader(
            data_values,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=shuffle,
        )

        n_neighbors = get_neighbors(featurized_db)

        return dataloader_db, n_neighbors.mean()

    def get_optimizer_scheduler_loss(
        self,
        optimizer_class=torch.optim.AdamW,
        scheduler_class=torch.optim.lr_scheduler.ExponentialLR,
        scheduler_settings: dict[str, Any] = dict(gamma=0.96),
        loss_function=torch.nn.L1Loss,
    ):
        """
        Configure the optimizer, learning rate scheduler, and loss function for training.

        This method sets up the components required for model training, including the optimizer,
        learning rate scheduler, and loss function. The optimizer and scheduler are configured
        based on the provided class arguments, while the loss function is selected based on
        the callable function provided.

        Args:
            optimizer_class (torch.optim.Optimizer, optional): The optimizer class to use.
                                                            Defaults to `torch.optim.AdamW`.
            scheduler_class (torch.optim.lr_scheduler._LRScheduler, optional): The learning rate
                                                                scheduler class to use.
                                                                Defaults to `torch.optim.lr_scheduler.ExponentialLR`.
            scheduler_settings (dict, optional): A dictionary of settings for the learning rate scheduler.
                                                For example, `gamma` can be set to control the decay rate.
                                                Defaults to `{"gamma": 0.96}`.
            loss_function (callable, optional): The loss function to use for training.
                                                Defaults to `torch.nn.L1Loss`.

        Raises:
            ValueError: If `optimizer_class` is not a subclass of `torch.optim.Optimizer`.
            ValueError: If `scheduler_class` is not a subclass of `torch.optim.lr_scheduler._LRScheduler`.
            ValueError: If `loss_function` is not callable.
        """
        if not issubclass(optimizer_class, torch.optim.Optimizer):
            raise ValueError("optimizer_class must be a subclass of torch.optim.Optimizer.")
        if not issubclass(scheduler_class, torch.optim.lr_scheduler.LRScheduler):
            raise ValueError("scheduler_class must be a subclass of torch.optim.lr_scheduler.")
        if not callable(loss_function):
            raise ValueError("loss_function must be callable.")

        self.optimizer = optimizer_class
        self.scheduler = scheduler_class
        self.loss_function = loss_function
        self._scheduler_settings = scheduler_settings
        logger.info(f"Optimizer configured: {optimizer_class.__name__}")
        logger.info(f"Learning rate scheduler configured: {scheduler_class.__name__}")
        logger.info(f"Loss function configured: {loss_function.__name__}")

    def _build_regressor(self, num_neighbors: float):
        """
        Build and initialize the regressor models using specified settings.

        This method constructs a regressor model for each committer and stores the models
        in the `self.models` dictionary. Each model is initialized with unique settings,
        and the model configuration is saved as a JSON file for reproducibility. The
        models are built using a `PeriodicNetwork` with predefined settings, including atom
        embeddings, number of layers, and other hyperparameters.

        Args:
            num_neighbors (float): Scaling factor based on the typical number of neighbors
                                for the convolution operation. It influences the model's
                                sensitivity to local atomic environments.
        """
        out_dim = 1  # Predict a scalar output
        em_dim = 64

        models = {}

        for i in tqdm(range(self.n_committers), desc="models"):
            model_settings_path = self.models_dir / (self._model_spec + f".rep{i}.json")
            model_settings = dict(
                in_dim=118,  # dimension of one-hot encoding of atom type
                em_dim=em_dim,  # dimension of atom-type embedding
                irreps_in=str(em_dim)
                + "x0e",  # em_dim scalars (L=0 and even parity) on each atom to represent atom type
                irreps_out=str(out_dim) + "x0e",  # out_dim scalars (L=0 and even parity) to output
                irreps_node_attr=str(em_dim)
                + "x0e",  # em_dim scalars (L=0 and even parity) on each atom to represent atom type
                layers=self.conv_layers,  # number of nonlinearities (number of convolutions = layers + 1)
                mul=32,  # multiplicity of irreducible representations
                lmax=self.l_max,  # maximum order of spherical harmonics
                max_radius=self.r_max,  # cutoff radius for convolution
                num_neighbors=num_neighbors,  # scaling factor based on the typical number of neighbors
                reduce_output=True,  # whether or not to aggregate features of all atoms at the end
            )
            with open(model_settings_path, "w") as fp:
                json.dump(model_settings, fp)
            models[f"model_{i}"] = PeriodicNetwork(**model_settings)
        self.models: dict[str, torch.nn.Module] = models

    def compile_(
        self,
        num_neighbors: float,
        lr: float = DEFAULT_OPTIM_SETTINGS["lr"],
        wd: float = DEFAULT_OPTIM_SETTINGS["wd"],
        optimizer_class=torch.optim.AdamW,
        scheduler_class=torch.optim.lr_scheduler.ExponentialLR,
        scheduler_settings: dict[str, Any] = dict(gamma=0.99),
        loss_function=torch.nn.L1Loss,
    ):
        """
        Compile and configure the model for training, setting up necessary components
        such as the optimizer, learning rate scheduler, and loss function, and then
        builds the regressors.

        This method performs the following steps:
        1. Loads the optimizer settings (learning rate and weight decay).
        2. Configures the optimizer, learning rate scheduler, and loss function.
        3. Builds the regressor models based on the provided number of neighbors.

        Args:
            num_neighbors (float): The scaling factor based on the typical number of neighbors.
            lr (float, optional): The learning rate for the optimizer.
                                    Defaults to `DEFAULT_OPTIM_SETTINGS["lr"]`.
            wd (float, optional): The weight decay for the optimizer.
                                    Defaults to `DEFAULT_OPTIM_SETTINGS["wd"]`.
            optimizer_class (torch.optim.Optimizer, optional): The optimizer class to use.
                                                                Defaults to `torch.optim.AdamW`.
            scheduler_class (torch.optim.lr_scheduler._LRScheduler, optional): The learning rate scheduler class.
                                                                Defaults to `torch.optim.lr_scheduler.ExponentialLR`.
            scheduler_settings (dict, optional): The settings for the learning rate scheduler.
                                                    Defaults to `dict(gamma=0.99)`.
            loss_function (torch.nn.Module, optional): The loss function to use for training.
                                                        Defaults to `torch.nn.L1Loss`.

        Raises:
            ValueError: If `num_neighbors` is not a positive number.
        """
        if num_neighbors <= 0:
            raise ValueError("'num_neighbors' must be a positive number.")

        logger.info("[STEP 1] Loading settings")
        self.set_optimizer_settings(lr, wd)

        logger.info("[STEP 2] Configuring optimizer, lr_scheduler, and loss function for training")
        self.get_optimizer_scheduler_loss(
            optimizer_class,
            scheduler_class,
            scheduler_settings,
            loss_function,
        )

        logger.info("[STEP 3] Building regressors")
        self._build_regressor(num_neighbors)

    def fit(
        self,
        dataloader_train: DataLoader,
        dataloader_valid: DataLoader,
        n_epochs: int = DEFAULT_TRAINING_SETTINGS["n_epochs"],
        parallelize: bool = False,
    ):
        """
        Train the regressor models using the specified training and validation datasets.

        This method supports both sequential and parallelized training:

        - If `parallelize` is `True` and multiple GPUs are available, training is executed asynchronously.
        - If only one GPU is available, a warning is issued, and sequential training is used.
        - If no GPUs are available but the model is set to use CUDA, an error is raised.
        - Otherwise, models are trained sequentially.

        Args:
            dataloader_train (DataLoader): PyTorch DataLoader containing the training dataset.
            dataloader_valid (DataLoader): PyTorch DataLoader containing the validation dataset.
            n_epochs (int, optional): Number of training epochs. Defaults to `DEFAULT_TRAINING_SETTINGS["n_epochs"]`.
            parallelize (bool, optional): Whether to parallelize training across multiple GPUs. Defaults to `False`.

        Raises:
            RuntimeError: If CUDA is selected but no GPU is available.
        """
        self.set_training_settings(n_epochs)

        if parallelize and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self._run_async(dataloader_train, dataloader_valid)
        else:
            if parallelize and torch.cuda.is_available():
                logger.warning("Only one GPU available, the flag 'parallelize' is ignored.")
            elif not torch.cuda.is_available() and "cuda" in self.device:
                logger.error(
                    f"You are attempting to run the training on {self.device} but only CPUs are available, check your drivers' installation or model settings!"
                )
                raise RuntimeError(
                    f"You are attempting to run the training on {self.device} but only CPUs are available, check your drivers' installation or model settings!"
                )
            for i in range(self.n_committers):
                _train_model(
                    i,
                    self.device,
                    self._model_spec,
                    self.models[f"model_{i}"],
                    self.models_dir,
                    self.optimizer,
                    self.learning_rate,
                    self.weight_decay,
                    self.scheduler,
                    self._scheduler_settings,
                    self.loss_function,
                    self.n_epochs,
                    dataloader_train,
                    dataloader_valid,
                )

    # Run training asynchronously
    async def _multi(self, dataloader_train: DataLoader, dataloader_valid: DataLoader):
        """
        Run training asynchronously for multiple models.

        This method launches multiple training processes asynchronously,
        distributing the models across available GPU devices.

        Args:
            dataloader_train (DataLoader): Training dataset wrapped in a PyTorch DataLoader.
            dataloader_valid (DataLoader): Validation dataset wrapped in a PyTorch DataLoader.

        Raises:
            RuntimeError: If CUDA devices are not available but GPU training is attempted.

        Note:
            - Each model is assigned to a CUDA device in a round-robin fashion.
            - Uses asyncio to manage multiple concurrent training executions.
        """
        logger.info("[ASYNC] Launching training asynchronously...")
        loop = asyncio.get_event_loop()
        tasks = []

        for i in range(self.n_committers):
            logger.info(
                f"[ASYNC] Iteration {i+1}/{self.n_committers}: Launching training on cuda:{i}..."
            )
            task = loop.run_in_executor(
                None,
                _train_model,
                i,
                f"cuda:{i % torch.cuda.device_count()}",
                self._model_spec,
                self.models[f"model_{i}"],
                self.models_dir,
                self.optimizer,
                self.learning_rate,
                self.weight_decay,
                self.scheduler,
                self._scheduler_settings,
                self.loss_function,
                self.n_epochs,
                dataloader_train,
                dataloader_valid,
            )
            tasks.append(task)
        await asyncio.gather(*tasks)
        logger.info("[ASYNC] All training processes completed!")

    def _run_async(self, dataloader_train: DataLoader, dataloader_valid: DataLoader):
        """
        Execute the asynchronous training routine based on the execution environment.

        This method ensures compatibility between Jupyter notebooks and standalone scripts
        when running the `_multi` training process. It checks for an active event loop:

        - If running inside a Jupyter notebook, it schedules `_multi` as a background task.
        - If running in a script, it uses `asyncio.run()` to properly start the event loop.

        Args:
            dataloader_train: The training dataset wrapped in a PyTorch DataLoader.
            dataloader_valid: The validation dataset wrapped in a PyTorch DataLoader.

        Returns:
            If running in a Jupyter notebook, returns an `asyncio.Task` object.
            If running as a script, the method executes `_multi` synchronously and returns `None`.
        """
        try:
            if asyncio.get_running_loop():  # Running inside Jupyter
                return asyncio.create_task(
                    self._multi(dataloader_train, dataloader_valid)
                )  # Run as a background task
        except RuntimeError:  # Running as a script
            logger.info("[SCRIPT] Running in a script. Using asyncio.run().")
            asyncio.run(self._multi(dataloader_train, dataloader_valid))

    def plot_history(self):
        """
        Plot the training and validation loss history for each trained model.

        This method iterates through the models saved in the model directory,
        loads their training history, and generates a plot comparing training and
        validation loss over epochs. The plot is saved as both PNG and PDF files
        in the figures directory.

        The plot will include:
            - X-axis: Epochs (steps)
            - Y-axis: Loss values
            - Two lines: Training loss and Validation loss

        Saves the generated plots as:
            - model_name_training.png
            - model_name_training.pdf

        Uses Matplotlib to generate the plots and saves them in the configured figures directory.

        Raises:
            FileNotFoundError: If no models are found in the specified model directory.
            KeyError: If the model history does not contain expected keys like "history".
        """
        for f in os.listdir(self.models_dir):
            if f.endswith(".torch"):
                model_history_path = self.models_dir / f
                try:
                    # Load history
                    history = torch.load(
                        model_history_path, map_location=self.device, weights_only=True
                    )["history"]
                except KeyError:
                    logger.error(f"KeyError: 'history' not found in {model_history_path}")
                    raise KeyError(
                        f"Model history for {f} does not contain expected 'history' key."
                    )

                # If history is loaded, set flag to True
                models_found = True

                steps = [d["step"] + 1 for d in history]
                loss_train = [d["train"]["loss"] for d in history]
                loss_valid = [d["valid"]["loss"] for d in history]

                fig, ax = plt.subplots(figsize=(7, 3), dpi=150)
                ax.plot(steps, loss_train, "o-", label="Training")
                ax.plot(steps, loss_valid, "o-", label="Validation")
                ax.set_xlabel("epochs")
                ax.set_ylabel("loss")
                ax.legend(frameon=False)
                fig.savefig(
                    self.figures_dir / f.replace(".torch", "_training.png"),
                    dpi=330,
                    bbox_inches="tight",
                )
                fig.savefig(
                    self.figures_dir / f.replace(".torch", "_training.pdf"),
                    dpi=330,
                    bbox_inches="tight",
                )
                plt.show()

        # Raise error if no models are found
        if not models_found:
            logger.error("No models found in the specified model directory.")
            raise FileNotFoundError("No model files found in the models directory.")

    def evaluate(self, dataloader: DataLoader, return_df: bool = False):
        """
        Evaluate the performance of the regression model(s) on the provided dataset.

        This method runs inference on the given dataset and calculates the loss
        (L1 loss) for each model in the `self.models` list. It returns either a
        detailed DataFrame with predictions and losses or a dictionary of predictions
        for each model, depending on the `return_df` flag.

        Args:
            dataloader (DataLoader): The DataLoader object containing the dataset
                to be evaluated.
            return_df (bool): Whether to return the results as a DataFrame with
                predictions and losses (`True`), or a dictionary with per-model
                results (`False`). Default is `False`.

        Returns:
            (pd.DataFrame): If `return_df=True`, returns a pandas DataFrame where each column
                            corresponds to predictions and loss for each model.
                            The columns include:
                                - `true_value`: Ground truth values.
                                - `model_i_prediction`: Predictions from model `i`.
                                - `model_i_loss`: L1 loss for model `i`.

            (dict[str, pd.DataFrame]): If `return_df=False`, returns a dictionary where each key is
                    a model identifier (e.g., `model_0`, `model_1`, ...) and the value is a
                    DataFrame containing the following columns:
                        - `true_value`: Ground truth values.
                        - `prediction`: Predictions from the model.
                        - `loss`: L1 loss computed for each sample.
        """
        if return_df:
            prediction_nn = pd.DataFrame()
            prediction_nn["true_value"] = [
                item[0] for batch in dataloader for item in batch["target"].tolist()
            ]

            for i in tqdm(range(self.n_committers), desc="models"):
                prediction_nn[f"model_{i}_prediction"] = np.empty(
                    (len(dataloader.dataset), 1)
                ).tolist()
                prediction_nn[f"model_{i}_loss"] = 0.0

                self.models[f"model_{i}"].to(self.device)
                self.models[f"model_{i}"].eval()
                with torch.no_grad():
                    i0 = 0
                    for j, d in tqdm(enumerate(dataloader), total=len(dataloader)):
                        d.to(self.device)
                        output = self.models[f"model_{i}"](d)
                        loss = (
                            F.l1_loss(output, d.target, reduction="none")
                            .mean(dim=-1)
                            .cpu()
                            .numpy()
                        )
                        prediction_nn.loc[i0 : i0 + len(d.target) - 1, f"model_{i}_prediction"] = [
                            k for k in output.cpu().numpy()
                        ]
                        prediction_nn.loc[i0 : i0 + len(d.target) - 1, f"model_{i}_loss"] = loss
                        i0 += len(d.target)

        else:
            prediction_nn = {}
            for i in tqdm(range(self.n_committers), desc="models"):
                prediction_nn[f"model_{i}"] = pd.DataFrame()
                prediction_nn[f"model_{i}"]["loss"] = 0.0
                prediction_nn[f"model_{i}"]["true_value"] = [
                    item[0] for batch in dataloader for item in batch["target"].tolist()
                ]
                prediction_nn[f"model_{i}"]["prediction"] = np.empty(
                    (len(dataloader.dataset), 1)
                ).tolist()

                self.models[f"model_{i}"].to(self.device)
                self.models[f"model_{i}"].eval()
                with torch.no_grad():
                    i0 = 0
                    for j, d in tqdm(enumerate(dataloader), total=len(dataloader)):
                        d.to(self.device)
                        output = self.models[f"model_{i}"](d)
                        loss = (
                            F.l1_loss(output, d.target, reduction="none")
                            .mean(dim=-1)
                            .cpu()
                            .numpy()
                        )
                        prediction_nn[f"model_{i}"].loc[
                            i0 : i0 + len(d.target) - 1, "prediction"
                        ] = [k for k in output.cpu().numpy()]
                        prediction_nn[f"model_{i}"].loc[i0 : i0 + len(d.target) - 1, "loss"] = loss
                        i0 += len(d.target)

        return prediction_nn

    def plot_parity(
        self, predictions_dict: dict[str, pd.DataFrame], include_ensemble: bool = True
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
        """
        all_predictions = []
        fig, ax = plt.subplots(figsize=(5, 3), dpi=300)

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
            ax.axline(
                (np.mean(y_true), np.mean(y_true)), slope=1, lw=0.85, ls="--", color="k", zorder=2
            )

            # Add inset histogram
            if i == 0:  # Create inset only once
                axin = ax.inset_axes([0.65, 0.17, 0.3, 0.3])
            axin.hist(
                error, bins=int(np.sqrt(len(error))), alpha=0.6, color=colors[i % len(colors)]
            )
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
        fig.savefig(
            self.figures_dir / (self._model_spec + "_parity.png"), dpi=330, bbox_inches="tight"
        )
        fig.savefig(
            self.figures_dir / (self._model_spec + "_parity.pdf"), dpi=330, bbox_inches="tight"
        )

        # Show plot
        plt.show()

    def _evaluate_unknown(
        self,
        dataloader: DataLoader,
    ) -> pd.DataFrame:
        """
        Predicts the target property for candidate specialized materials using regressor models.

        Args:
            dataloader (DataLoader): A PyTorch DataLoader containing the dataset to be evaluated.

        Returns:
            (pd.DataFrame): A DataFrame containing the predictions and uncertainties for the dataset,
                            with one column for each regressor model and additional columns for
                            the mean (`regressor_mean`) and standard deviation (`regressor_std`)
                            of the predictions across all models.
        """
        prediction_nn = pd.DataFrame()

        for i in tqdm(range(self.n_committers), desc="models"):
            prediction_nn[f"regressor_{i}"] = np.empty((len(dataloader.dataset), 1)).tolist()

            self.models[f"model_{i}"].to(self.device)
            self.models[f"model_{i}"].eval()
            with torch.no_grad():
                i0 = 0
                for j, d in tqdm(enumerate(dataloader), total=len(dataloader)):
                    d.to(self.device)
                    output = self.models[f"model_{i}"](d)
                    # print([[k] for k in output.cpu().numpy()])
                    prediction_nn.loc[i0 : i0 + len(d.symbol) - 1, f"regressor_{i}"] = [
                        float(k) for k in output.cpu().numpy()
                    ]
                    i0 += len(d.symbol)

        prediction_nn["regressor_mean"] = prediction_nn[
            [f"regressor_{i}" for i in range(self.n_committers)]
        ].mean(axis=1)
        prediction_nn["regressor_std"] = prediction_nn[
            [f"regressor_{i}" for i in range(self.n_committers)]
        ].std(axis=1)

        return prediction_nn

    def predict(self, db: GNoMEDatabase, confidence_threshold: float = 0.5, save_final=True):
        """
        Predicts the target property for candidate specialized materials using regressor models,
        after filtering materials based on classifier committee confidence.

        Args:
            db (GNoMEDatabase): The database containing the materials and their properties.
            confidence_threshold (float, optional): The minimum classifier committee confidence
                                                    required to keep a material for prediction.
                                                    Defaults to `0.5`.
            save_final (bool, optional): Whether to save the final database with predictions.
                                        Defaults to `True`.

        Returns:
            (pd.DataFrame): A DataFrame containing the predictions, along with the true values and
                        classifier committee confidence scores for the screened materials.

        Notes:
            - The method filters the materials based on the classifier confidence, then uses the
            regressor models to predict the target property for the remaining materials.
            - If `save_final` is set to True, the predictions are saved to the database in the
            `final` stage.
        """
        logger.info(
            f"Discarding materials with classifier committee confidence threshold < {confidence_threshold}."
        )
        logger.info("Featurizing and loading database as `tg.loader.DataLoader`.")
        dataloader_db, _ = self.create_dataloader(db, confidence_threshold)
        logger.info("Predicting the target property for candidate specialized materials.")
        predictions = self._evaluate_unknown(dataloader_db)
        df = db.get_database("processed")
        screened = df[df["classifier_mean"] > confidence_threshold]
        predictions = pd.concat([screened.reset_index(), predictions], axis=1)

        if save_final:
            logger.info("Saving the final database.")
            db.databases["final"] = predictions.copy()
            db.save_database("final")

        return predictions
