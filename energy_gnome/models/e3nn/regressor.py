import asyncio
import glob
import json
import os
from pathlib import Path
import time
from typing import Any

from ase.io import read
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch_geometric as tg
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
    build_manager_dataloader,
    get_encoding,
    get_neighbors,
    train_regressor,
)
from energy_gnome.utils.readers import load_json, load_yaml, save_yaml, to_unix

DEFAULT_DTYPE = torch.float64
# BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"
# tqdm.pandas(bar_format=BAR_FORMAT)
tqdm.pandas()
torch.set_default_dtype(DEFAULT_DTYPE)
mp.set_start_method("spawn", force=True)


def train_model(
    i,
    device,
    model_spec,
    model,
    models_dir,
    optimizer,
    learning_rate,
    weight_decay,
    scheduler,
    scheduler_settings,
    loss_function,
    n_epochs,
    dataloader_train,
    dataloader_valid,
):
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
        Initialize the E3NNRegressor with root data and models directories.

        Sets up the e3nn regressor structure for building and training the regressor models.

        Args:
            model_name (str): ...
            models_dir (Path, optional): Root directory path for storing trained models.
                                         Defaults to MODELS_DIR from config.
        """
        self._model_spec = (
            "model.e3nn_regressor."
            + target_property  # + "." + time.strftime("%y%m%d", time.localtime())
        )
        self._manager = mp.Manager()
        super().__init__(
            model_name=model_name,
            target_property=target_property,
            models_dir=models_dir,
            figures_dir=figures_dir,
        )
        self.l_max: int = 2  #

    def _find_model_states(self):
        models_states = []
        if any(self.models_dir.iterdir()):
            models_states = [f_ for f_ in self.models_dir.iterdir() if f_.match("*.torch")]
        return models_states

    def set_model_settings(self, yaml_file: Path | str | None = None, **kargs):
        # Accessing model settings (YAML FILE)
        if yaml_file:
            self._load_model_setting(yaml_file)

        # Accessing model settings (DEFAULT)
        for att, defvalue in DEFAULT_E3NN_SETTINGS.items():
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

    def _save_model_settings(self):
        settigns_path = self.models_dir / (self._model_spec + ".yaml")
        settings = {}
        for att, _ in DEFAULT_E3NN_SETTINGS.items():
            settings[att] = getattr(self, att)
        logger.info(f"saving models 'general' settings in {settigns_path}")
        save_yaml(settings, settigns_path)

    def _load_model_setting(self, yaml_path):
        settings = load_yaml(yaml_path)
        for att, _ in DEFAULT_E3NN_SETTINGS.items():
            setattr(self, att, settings[att])

        if "cuda" in self.device and not torch.cuda.is_available():
            logger.warning(f"Models trained on {self.device} but only the cpu is available")
            self.device = "cpu"

    def set_training_settings(self, n_epochs: int):
        self.n_epochs = n_epochs

    def set_optimizer_settings(self, lr: float, wd: float):
        self.learning_rate = lr
        self.weight_decay = wd

    '''
    def db_featurizer(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Load and featurize the specified database.

        Checks for the presence of an existing database file for the given stage
        and loads it into a pandas DataFrame. If the database file does not exist,
        logs a warning and returns an empty DataFrame.
        Builds a pandas DataFrame ready for model training.

        Args:
            dataset (pd.DataFrame) = ...

        Returns:
            pd.DataFrame: The built database or an empty DataFrame if not found.
        """
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

        for i in tqdm(range(len(dataset))):
            database_nn.loc[i, self.target_property] = dataset[self.target_property].iloc[i]
            path = dataset["cif_path"].iloc[i]
            structure = read(Path(to_unix(path)))
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
    '''

    def featurize_db(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Load and featurize the specified database.

        Args:
            dataset (pd.DataFrame): Input dataset containing CIF file paths.

        Returns:
            pd.DataFrame: The featurized dataset ready for model training.
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
        confidence_threshold: float = 0.5,
    ):
        """
        Format dataloaders for PyTorch operations.

        Args:
            dataset (pd.DataFrame) = Featurized dataset.

        Returns:
            ???
        """
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

        featurized_db = self.featurize_db(df)

        data_values = featurized_db["data"].values
        dataloader_db = DataLoader(
            data_values, batch_size=self.batch_size, num_workers=0, pin_memory=True
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

    def build_regressor(self, num_neighbors: float):
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

    def compile(
        self,
        num_neighbors: float,
        lr: float = DEFAULT_OPTIM_SETTINGS["lr"],
        wd: float = DEFAULT_OPTIM_SETTINGS["wd"],
        optimizer_class=torch.optim.AdamW,
        scheduler_class=torch.optim.lr_scheduler.ExponentialLR,
        scheduler_settings: dict[str, Any] = dict(gamma=0.99),
        loss_function=torch.nn.L1Loss,
    ):
        # if not isinstance(databases, list):
        #     databases=[databases]

        logger.info("[STEP 1] Loading settings")
        self.set_optimizer_settings(lr, wd)

        # logger.info("[STEP 2] Featurize and format subsets for training")
        # training_db_list = []
        # valid_db_list = []
        # for db in databases:
        #     training_db_list.append(db.load_regressor_data("training"))
        #     valid_db_list.append(db.load_regressor_data("validation"))

        # if len(training_db_list) > 1 :
        #     training_db = pd.concat(training_db_list, ignore_index=True)
        #     valid_db = pd.concat(valid_db_list, ignore_index=True)
        # else:
        #     training_db = training_db_list[0]
        #     valid_db = valid_db_list[0]

        # train_feat = self.db_featurizer(training_db)
        # valid_feat = self.db_featurizer(valid_db)
        # dataloader_train, n_train = self.dataloader(train_feat)
        # dataloader_valid, _ = self.dataloader(valid_feat)

        logger.info(
            "[STEP 2] Configure the optimizer, lr_scheduler, and loss function for training"
        )
        self.get_optimizer_scheduler_loss(
            optimizer_class,
            scheduler_class,
            scheduler_settings,
            loss_function,
        )

        logger.info("[STEP 3] Build regressors")
        self.build_regressor(num_neighbors)

    def fit(
        self,
        dataloader_train: tg.loader.DataLoader,
        dataloader_valid: tg.loader.DataLoader,
        n_epochs: int = DEFAULT_TRAINING_SETTINGS["n_epochs"],
        parallelize: bool = False,
    ):
        self.set_training_settings(n_epochs)

        if parallelize and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.run_async(dataloader_train, dataloader_valid)

            """
            processes: list[mp.Process] = []
            n_gpus = torch.cuda.device_count()
            logger.debug(f"Attempting to run {self.n_committers} models on {n_gpus} GPUs")
            if n_gpus > self.n_committers:
                logger.warning(f"Training {self.n_committers}, but {n_gpus} GPUs are available, WASTE OF RESOURCES!")
            elif n_gpus < self.n_committers:
                logger.warning(f"Training {self.n_committers}, but {n_gpus} GPUs are available, WASTE OF PERFORMANCES!")
            for i in range(self.n_committers):
                device = f"cuda:{i % torch.cuda.device_count()}"
                logger.info(f"Initializing training model_{i} on device {device}")

                p = mp.Process(
                    target=train_model,
                    args=(i, device, self._model_spec, self.models, self.models_dir,
                          self.optimizer, self.learning_rate, self.weight_decay,
                          self.scheduler, self._scheduler_settings, self.loss_function,
                          self.n_epochs, dataloader_train, dataloader_valid)
                )
                processes.append(p)

            for i, p in enumerate(processes):
                device = f"cuda:{i % torch.cuda.device_count()}"
                logger.info(f"Starting training model_{i}")
                p.start()
                logger.debug(f"Process {p}, device {device}")

            for i, p in enumerate(processes):
                p.join()

            n_fail = 0
            for i, p in enumerate(processes):
                exitcode = p.exitcode
                if exitcode != 0:
                    logger.debug(f"Process {p} ({i}), FAILED!!! (exitcode={exitcode})")
                    n_fail += 1
                else:
                    logger.debug(f"Process {p} ({i}), ENDED!!! (exitcode={exitcode})")

            if n_fail > 0:
                raise RuntimeError(f"{n_fail} training processes failed!")
            """

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
                train_model(
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
    async def multi(self, dataloader_train, dataloader_valid):
        logger.info("[ASYNC] Launching training asynchronously...")
        loop = asyncio.get_event_loop()
        tasks = []

        for i in range(self.n_committers):
            logger.info(
                f"[ASYNC] Iteration {i+1}/{self.n_committers}: Launching training on cuda:{i}..."
            )
            task = loop.run_in_executor(
                None,
                train_model,
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

    def run_async(self, dataloader_train, dataloader_valid):
        """Handle running asyncio code in both scripts and Jupyter notebooks."""
        try:
            if asyncio.get_running_loop():  # Running inside Jupyter
                return asyncio.create_task(
                    self.multi(dataloader_train, dataloader_valid)
                )  # Run as a background task
        except RuntimeError:  # Running as a script
            logger.info("[SCRIPT] Running in a script. Using asyncio.run().")
            asyncio.run(self.multi(dataloader_train, dataloader_valid))

    def load_trained_models(self, state: str = "state_best"):
        for f in os.listdir(self.models_dir):
            if f.endswith(".yaml") and self._model_spec in f:
                yaml_path = self.models_dir / f
                self.set_model_settings(yaml_file=yaml_path)
        i = 0
        for f in os.listdir(self.models_dir):
            if f.endswith(".torch") and self._model_spec in f:
                model_setting_path = self.models_dir / f.replace(".torch", ".json")
                model_history_path = self.models_dir / f
                logger.info(f"Loading model with setting in {model_setting_path}")
                logger.info(f"And weights in {model_history_path}")
                model_setting = load_json(model_setting_path)
                self.models[f"model_{i}"] = PeriodicNetwork(**model_setting)
                self.models[f"model_{i}"].load_state_dict(
                    torch.load(model_history_path, map_location=self.device, weights_only=True)[
                        state
                    ]
                )
                self.models[f"model_{i}"].pool = True
                i += 1

        return [f for f in os.listdir(self.models_dir) if f.endswith(".torch")]

    def plot_history(self):
        for f in os.listdir(self.models_dir):
            if f.endswith(".torch"):
                model_history_path = self.models_dir / f
                history = torch.load(
                    model_history_path, map_location=self.device, weights_only=True
                )["history"]
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

    def evaluate(self, dataloader: tg.loader.DataLoader, return_df: bool = False):
        """
        Evaluate regressor performance.

        Args:
            dataset (pd.DataFrame) = Featurized dataset.

        Returns:
            ???
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
                        # print([[k] for k in output.cpu().numpy()])
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
                        # print([[k] for k in output.cpu().numpy()])
                        prediction_nn[f"model_{i}"].loc[
                            i0 : i0 + len(d.target) - 1, "prediction"
                        ] = [k for k in output.cpu().numpy()]
                        prediction_nn[f"model_{i}"].loc[i0 : i0 + len(d.target) - 1, "loss"] = loss
                        i0 += len(d.target)

        return prediction_nn

    def plot_parity(
        self, predictions_dict: dict[str, pd.DataFrame], include_ensemble: bool = True
    ):
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
