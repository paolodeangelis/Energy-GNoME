import json
from pathlib import Path
import time
from typing import Any

from ase.io import read
from loguru import logger
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric as tg
from tqdm import tqdm

from energy_gnome.config import (
    DEFAULT_E3NN_SETTINGS,
    DEFAULT_E3NN_TRAINING_SETTINGS,
    DEFAULT_OPTIM_SETTINGS,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)
from energy_gnome.models.regressor.base_regressor_model import BaseRegressor

# from energy_gnome.tools.model import PeriodicNetwork, train_regressor
from energy_gnome.models.regressor.e3nn_regressor_utils import (
    PeriodicNetwork,
    train_regressor,
)
from energy_gnome.tools.postprocessing import get_neighbors
from energy_gnome.tools.preprocessing import (
    build_data,
    get_encoding,
    train_valid_test_split,
)

DEFAULT_DTYPE = torch.float64
BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"
tqdm.pandas(bar_format=BAR_FORMAT)
torch.set_default_dtype(DEFAULT_DTYPE)


class E3NNRegressor(BaseRegressor):
    def __init__(
        self,
        data_dir: Path | str = PROCESSED_DATA_DIR,
        models_dir: Path | str = MODELS_DIR,
    ):
        """
        Initialize the E3NNRegressor with root data and models directories.

        Sets up the e3nn regressor structure for building and training the regressor models.

        Args:
            data_dir (Path, optional): Root directory path for reading data.
                                       Defaults to PROCESSED_DATA_DIR from config.
            models_dir (Path, optional): Root directory path for storing trained models.
                                         Defaults to MODELS_DIR from config.

        Raises:
            NotImplementedError: If the specified material class is not supported.
        """
        super().__init__(data_dir=data_dir, models_dir=models_dir)

    def model_settings(
        self,
        category: str = "perovskites",
        target_property: str = "band_gap",
        settings: dict = DEFAULT_E3NN_SETTINGS,
    ):
        self.data_dir: str | Path = Path(PROCESSED_DATA_DIR, category)
        self.models_dir = Path(MODELS_DIR, category)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.target_property = target_property

        # Accessing model settings
        self.n_committers = settings["n_committers"]
        self.l_max = settings["l_max"]
        self.r_max = settings["r_max"]
        self.device = settings["device"]
        self.model_name = (
            "model." + target_property + "." + time.strftime("%y%m%d", time.localtime())
        )

    def training_settings(self, settings: dict = DEFAULT_E3NN_TRAINING_SETTINGS):
        # Accessing training settings
        self.n_epochs = settings["n_epochs"]
        self.batch_size = settings["batch_size"]
        self.load_db = settings["load_db"]

    def optimizer_settings(self, settings: dict = DEFAULT_OPTIM_SETTINGS):
        self.learning_rate = settings["lr"]
        self.weight_decay = settings["wd"]

    def load_and_build_database(self) -> pd.DataFrame:
        """
        Load the existing processed database.

        Checks for the presence of an existing database file for the given category
        and loads it into a pandas DataFrame. If the database file does not exist,
        logs a warning and returns an empty DataFrame.
        Builds a pandas DataFrame ready for model training.

        Returns:
            pd.DataFrame: The built database or an empty DataFrame if not found.
        """
        db_path = self.data_dir
        if db_path.exists():
            datasets = pd.read_json(Path(db_path) / "database.json")
            logger.debug(f"Loaded existing database from {db_path}")
        else:
            logger.warning(f"No existing database found at {db_path}")

        database_nn = pd.DataFrame(
            {
                self.target_property: pd.Series(dtype="float"),
                "structure": pd.Series(dtype="object"),
                "formula": pd.Series(dtype="str"),
                "species": pd.Series(dtype="object"),
                "data": pd.Series(dtype="object"),
            }
        )

        pd.options.mode.chained_assignment = None  # default='warn', hides warning

        for i in tqdm(range(len(datasets))):
            database_nn.loc[i, self.target_property] = datasets[self.target_property].iloc[i]
            path = datasets["cif_path"].iloc[i]
            structure = read(path)
            database_nn.at[i, "structure"] = (
                structure.copy()
            )  # if not working, try removing .copy()
            database_nn.loc[i, "formula"] = structure.get_chemical_formula()
            database_nn.at[i, "species"] = list(set(structure.get_chemical_symbols()))

        type_encoding, type_onehot, atomicmass_onehot = get_encoding()

        r_max = self.r_max  # cutoff radius
        database_nn["data"] = database_nn.progress_apply(
            lambda x: build_data(
                x,
                type_encoding,
                type_onehot,
                atomicmass_onehot,
                self.target_property,
                r_max,
                dtype=DEFAULT_DTYPE,
            ),
            axis=1,
        )

        return database_nn

    def random_split(self, database_nn, valid_size=0.2, test_size=0.05, seed=42):
        """
        Perform a random train-validation-test split with specified sizes.


        Args:
            database_nn (pd.DataFrame): DataFrame to split.
            valid_size (float64, optional): Size of the validation set, as fraction of the whole starting DataFrame.
                                            Defaults to 0.2.
            test_size (float64, optional): Size of the testing set, as fraction of the whole starting DataFrame.
                                           Defaults to 0.05.
            seed (int, optional): Controls the shuffling applied to the data before applying the split.
                                  Pass an int for reproducible output across multiple function calls.
                                  Defaults to 42.

        Returns:
            dataloader_dict: dictionary of subset dataloader objects.
            idx_dict: dictionary of subset indices.
            n_dict: dictionary of subset sizes.
        """
        species = sorted(list(set(database_nn["species"].sum())))
        idx_train, idx_valid, idx_test = train_valid_test_split(
            database_nn, species, valid_size=valid_size, test_size=test_size, seed=seed, plot=False
        )
        idx_dict = {"train": idx_train, "valid": idx_valid, "test": idx_test}

        # format dataloaders
        dataloader_train = tg.loader.DataLoader(
            database_nn.iloc[idx_train]["data"].values, batch_size=self.batch_size
        )
        dataloader_valid = tg.loader.DataLoader(
            database_nn.iloc[idx_valid]["data"].values, batch_size=self.batch_size
        )
        dataloader_test = tg.loader.DataLoader(
            database_nn.iloc[idx_test]["data"].values, batch_size=self.batch_size
        )
        dataloader_dict = {
            "train": dataloader_train,
            "valid": dataloader_valid,
            "test": dataloader_test,
        }

        n_train = get_neighbors(database_nn, idx_train)
        n_valid = get_neighbors(database_nn, idx_valid)
        n_test = get_neighbors(database_nn, idx_test)
        n_dict = {"train": n_train, "valid": n_valid, "test": n_test}

        return dataloader_dict, idx_dict, n_dict

    def get_optimizer_scheduler_loss(
        self,
        optimizer_class=torch.optim.AdamW,
        scheduler_class=torch.optim.lr_scheduler.ExponentialLR,
        scheduler_settings: dict[str, Any] = dict(gamma=0.96),
        loss_function=torch.nn.L1Loss,
    ):
        """
        Configure the optimizer, lr_scheduler, and loss function for training.

        Args:
            optimizer_class: A PyTorch optimizer class. Defaults to torch.optim.AdamW.
            scheduler_class: A PyTorch lr_scheduler class. Defaults to torch.optim.lr_scheduler.ExponentialLR.
            loss_function: A PyTorch loss function. Defaults to torch.nn.L1Loss.

        Raises:
        ValueError: If optimizer_class is not a subclass of torch.optim.Optimizer.
        ValueError: If loss_function is not callable.
        ValueError: If scheduler_class is provided but not a subclass of torch.optim.lr_scheduler._LRScheduler.
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
        print(f"Optimizer configured: {optimizer_class.__name__}")
        print(f"Learning rate scheduler configured: {scheduler_class.__name__}")
        print(f"Loss function configured: {loss_function.__name__}")

    def build_regressor(self, n_train):
        out_dim = 1  # Predict a scalar output
        em_dim = 64

        models = {}

        for i in tqdm(range(self.n_committers), desc="models"):
            model_settings_path = self.models_dir / (self.model_name + f".rep{i}.json")
            model_settings = dict(
                in_dim=118,  # dimension of one-hot encoding of atom type
                em_dim=em_dim,  # dimension of atom-type embedding
                irreps_in=str(em_dim)
                + "x0e",  # em_dim scalars (L=0 and even parity) on each atom to represent atom type
                irreps_out=str(out_dim) + "x0e",  # out_dim scalars (L=0 and even parity) to output
                irreps_node_attr=str(em_dim)
                + "x0e",  # em_dim scalars (L=0 and even parity) on each atom to represent atom type
                layers=2,  # number of nonlinearities (number of convolutions = layers + 1)
                mul=32,  # multiplicity of irreducible representations
                lmax=self.l_max,  # maximum order of spherical harmonics
                max_radius=self.r_max,  # cutoff radius for convolution
                num_neighbors=n_train.mean(),  # scaling factor based on the typical number of neighbors
                reduce_output=True,  # whether or not to aggregate features of all atoms at the end
            )
            with open(model_settings_path, "w") as fp:
                json.dump(model_settings, fp)
            models[f"model_{i}"] = PeriodicNetwork(**model_settings)
        self.models: dict[str, torch.nn.Module] = models

    def train_regressor(
        self, dataloader_train: tg.loader.DataLoader, dataloader_valid: tg.loader.DataLoader
    ):
        for i in tqdm(range(self.n_committers), desc="models"):
            model_path = self.models_dir / (self.model_name + f".rep{i}")
            model_: torch.nn.Module = self.models[f"model_{i}"]
            model_.pool = True

            # Compile
            opt = self.optimizer(
                model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
            scheduler = self.scheduler(opt, **self._scheduler_settings)
            loss_fn = self.loss_function()

            train_regressor(
                model_,
                opt,
                dataloader_train,
                dataloader_valid,
                loss_fn,
                model_path,
                max_iter=self.n_epochs,  # number of iterations
                scheduler=scheduler,
                only_best=True,
                device=self.device,
            )

    def eval_regressor(self, database_nn, idx_train, idx_valid, idx_test):
        prediction_nn = {}
        for i in tqdm(range(self.n_committers), desc="models"):
            model_path = str(self.models_dir / self.model_name) + f".rep{i}.torch"
            self.models[f"model_{i}"].load_state_dict(
                torch.load(Path(model_path), map_location=self.device)["state_best"]
            )
            self.models[f"model_{i}"].pool = True

            dataloader = tg.loader.DataLoader(database_nn["data"].values, batch_size=4)
            prediction_nn[f"model_{i}"] = pd.DataFrame()
            prediction_nn[f"model_{i}"]["loss"] = 0.0
            prediction_nn[f"model_{i}"][self.target_property] = database_nn[self.target_property]
            prediction_nn[f"model_{i}"]["prediction"] = np.empty((len(database_nn), 1)).tolist()

            self.models[f"model_{i}"].to(self.device)
            self.models[f"model_{i}"].eval()
            with torch.no_grad():
                i0 = 0
                for j, d in tqdm(enumerate(dataloader), total=len(dataloader)):
                    d.to(self.device)
                    output = self.models[f"model_{i}"](d)
                    loss = (
                        F.mse_loss(output, d.target, reduction="none").mean(dim=-1).cpu().numpy()
                    )
                    # print([[k] for k in output.cpu().numpy()])
                    prediction_nn[f"model_{i}"].loc[i0 : i0 + len(d.target) - 1, "prediction"] = [
                        k for k in output.cpu().numpy()
                    ]
                    prediction_nn[f"model_{i}"].loc[i0 : i0 + len(d.target) - 1, "loss"] = loss
                    i0 += len(d.target)

        data = {}
        for i in range(self.n_regressors):
            data[f"model_{i}"] = {
                "train": prediction_nn[f"model_{i}"].iloc[idx_train],
                "valid": prediction_nn[f"model_{i}"].iloc[idx_valid],
                "test": prediction_nn[f"model_{i}"].iloc[idx_test],
            }
        return data

    """
    Inserire plot o altro modo per mostrare i risultati del testing
    """
