# parallel_training.py
# ---------------------
# This script demonstrates how to train an E(3) Equivariant Neural Network (E3NN) regressor
# using the energy_gnome framework with optional parallelization.

# The pipeline includes:
# 1. Data loading
# 2. Train/validation/test splitting
# 3. Structure-based graph feature generation
# 4. E3NN model training using ensemble learning (via multiple committers)
# 5. Evaluation and visualization

from pathlib import Path

from energy_gnome.dataset import PerovskiteDatabase
from energy_gnome.models import E3NNRegressor


def main():
    # -------------------------
    # Define project directories
    # -------------------------
    # Adjust paths to match your project structure.
    data_dir = Path(".").resolve().parent / "data"
    models_dir = Path(".").resolve().parent / "models"
    figures_dir = Path(".").resolve().parent / "figures"

    # ----------------------
    # Load prepared databases
    # ----------------------
    # Loads the perovskite dataset.
    perov_db = PerovskiteDatabase(name="perovskites", data_dir=data_dir)
    print(perov_db)

    # -------------------------------------
    # Create balanced train/val/test splits
    # -------------------------------------
    # Target property: band_gap
    # Stratified splitting ensures representative elemental and label distributions.
    perov_db.split_regressor(target_property="band_gap", valid_size=0.2, test_size=0.05, save_split=True)

    # -------------------------------
    # Initialize E(3)NN Regressor Model
    # -------------------------------
    # Sets model parameters and training device.
    regressor_model = E3NNRegressor(
        model_name="perov_e3nn",  # Naming may reflect target or dataset
        target_property="band_gap",
        models_dir=models_dir,
        figures_dir=figures_dir,
    )

    regressor_model.set_model_settings(
        n_committers=4,  # Train an ensemble of 4 models
        l_max=3,  # Maximum spherical harmonic degree
        r_max=5.0,  # Radius cutoff for neighbors
        conv_layers=2,  # Number of equivariant convolutional layers
        device="cuda:0",  # Set to 'cpu' or appropriate CUDA device
        batch_size=16,
    )

    # ----------------------
    # Graph Feature Engineering
    # ----------------------
    # Converts crystal structures into graph representations.
    # Returns dataloaders for each data split and the average neighbor count.
    train_dl, n_neigh_mean = regressor_model.create_dataloader(databases=[perov_db], subset="training")
    valid_dl, _ = regressor_model.create_dataloader(databases=[perov_db], subset="validation")
    test_dl, _ = regressor_model.create_dataloader(databases=[perov_db], subset="testing")

    # ------------------
    # Model Compilation
    # ------------------
    # Prepares the model for training by configuring neighbor settings and learning rate scheduler.
    regressor_model.compile(
        num_neighbors=n_neigh_mean, scheduler_settings={"gamma": 0.98}  # Learning rate decay factor
    )

    # --------------------------
    # Training with Parallelization
    # --------------------------
    # Trains all committers in parallel. This can speed up training significantly
    # on multi-GPU or multi-core systems. Requires multiprocessing-compatible environment.
    regressor_model.fit(dataloader_train=train_dl, dataloader_valid=valid_dl, n_epochs=5, parallelize=True)

    # ------------------
    # Training Diagnostics
    # ------------------
    # Visualize learning curves to monitor model convergence.
    regressor_model.plot_history()

    # --------------------------
    # Prediction & Evaluation
    # --------------------------
    # Run evaluation on training, validation, and test sets.
    # This returns prediction distributions from the ensemble.
    train_predictions = regressor_model.evaluate(dataloader=train_dl)
    valid_predictions = regressor_model.evaluate(dataloader=valid_dl)
    test_predictions = regressor_model.evaluate(dataloader=test_dl)

    # ------------------------
    # Parity Plot Visualization
    # ------------------------
    # Plot predicted vs. true values (with ensemble means).
    regressor_model.plot_parity(predictions_dict=train_predictions, include_ensemble=True)
    regressor_model.plot_parity(predictions_dict=valid_predictions, include_ensemble=True)
    regressor_model.plot_parity(predictions_dict=test_predictions, include_ensemble=True)


# Run the script
if __name__ == "__main__":
    main()
