"""source: https://github.com/ninarina12/phononDoS_tutorial ?
"""

from collections.abc import Generator
import copy
import math
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torcheval.metrics.functional import binary_accuracy, binary_auroc
from tqdm.auto import tqdm

BAR_FROMAT = "{l_bar}{bar:10}{r_bar}"


def loglinspace(rate: float, step: int, end: int | None = None) -> Generator[int, None, None]:
    """
    Generates a logarithmic spaced sequence of numbers.

    Parameters:
        rate (float): The rate at which the step size increases.
        step (int): The initial step size.
        end (Optional[int]): The maximum number to be generated. If None, the generator is infinite.

    Yields:
        int: The next number in the logarithmic sequence.
    """
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step * (1 - math.exp(-t * rate / step)))


def evaluate_regressor(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    # loss_fn_mae: torch.nn.Module,
    device: str,
) -> tuple[float, float]:
    """
    Evaluates the model on the given dataloader.

    Parameters:
        model (torch.nn.Module): The neural network model to be evaluated.
        dataloader (DataLoader): The DataLoader containing the data to evaluate the model on.
        loss_fn (torch.nn.Module): The loss function used for evaluation.
        loss_fn_mae (torch.nn.Module): The mean absolute error loss function.
        device (str): The device (e.g., 'cuda' or 'cpu') on which to perform the evaluation.

    Returns:
        Tuple[float, float]: The average loss and average mean absolute error across all batches in the dataloader.
    """
    model.eval()
    loss_cumulative = 0.0
    # loss_cumulative_mae = 0.0
    # start_time = time.time()
    with torch.no_grad():
        for j, d in enumerate(dataloader):
            d.to(device)
            output = model(d)
            loss = loss_fn(output, d.target).cpu()
            # loss_mae = loss_fn_mae(output, d.target).cpu()
            loss_cumulative = loss_cumulative + loss.detach().item()
            # loss_cumulative_mae = loss_cumulative_mae + loss_mae.detach().item()
    return loss_cumulative / len(dataloader)  # , loss_cumulative_mae / len(dataloader)


def train_regressor(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader_train: DataLoader,
    dataloader_valid: DataLoader,
    loss_fn: torch.nn.Module,
    # loss_fn_mae: torch.nn.Module,
    run_name: str,
    max_iter: int = 101,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    only_best: bool = True,
    device: str = "cpu",
) -> None:
    """
    Trains the model using the given dataloaders, optimizer, and loss functions.

    Parameters:
        model (torch.nn.Module): The neural network model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        dataloader_train (DataLoader): The DataLoader for training data.
        dataloader_valid (DataLoader): The DataLoader for validation data.
        loss_fn (torch.nn.Module): The loss function used for training.
        loss_fn_mae (torch.nn.Module): The mean absolute error loss function.
        run_name (str): The name of the run, used for saving model checkpoints.
        max_iter (int): Maximum number of training iterations.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler, optional.
        only_best (bool): Flag to save only the best model based on validation loss.
        device (str): The device (e.g., 'cuda' or 'cpu') on which to train the model.

    Saves:
        Model checkpoints and training history to a file named "{run_name}.torch".
    """
    model.to(device)

    checkpoint_generator = loglinspace(0.3, 5)
    checkpoint = next(checkpoint_generator)
    start_time = time.time()

    try:
        model.load_state_dict(torch.load(run_name + ".torch")["state"])
        best_model_state = copy.deepcopy(model.state_dict())
    except:  # noqa
        results = {}
        history = []
        s0 = 0
    else:
        results = torch.load(run_name + ".torch")
        history = results["history"]
        s0 = history[-1]["step"] + 1
    try:
        history[-1]["valid"]
    except (KeyError, IndexError):
        loss_valid_best = 9.9999e99
    else:
        loss_valid_best = min([d["valid"]["loss"] for d in history])

    for step in range(max_iter):
        model.train()
        loss_cumulative = 0.0
        # loss_cumulative_mae = 0.0

        for j, d in tqdm(
            enumerate(dataloader_train), total=len(dataloader_train), bar_format=BAR_FROMAT
        ):
            d.to(device)
            output = model(d)
            loss = loss_fn(output, d.target).cpu()
            # loss_mae = loss_fn_mae(output, d.target).cpu()
            loss_cumulative = loss_cumulative + loss.detach().item()
            # loss_cumulative_mae = loss_cumulative_mae + loss_mae.detach().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end_time = time.time()
        wall = end_time - start_time

        # save history

        evaluate_model = (step == checkpoint) or only_best
        # print(f"evaluate_model {evaluate_model} ({step} == {checkpoint} or {only_best})")
        if evaluate_model:
            eval_start_time = time.time()
            valid_avg_loss = evaluate_regressor(
                model,
                dataloader_valid,
                loss_fn,
                device,  # loss_fn_mae, device
            )
            # train_avg_loss = evaluate(model, dataloader_train, loss_fn, loss_fn_mae, device)
            train_avg_loss = loss_cumulative / len(
                dataloader_train
            )  # , loss_cumulative_mae / len(dataloader_train)

            history.append(
                {
                    "step": s0 + step,
                    "wall": wall,
                    "batch": {
                        "loss": loss.item(),
                        # "mean_abs": loss_mae.item(),
                    },
                    "valid": {
                        "loss": valid_avg_loss,
                        # "mean_abs": valid_avg_loss[1],
                    },
                    "train": {
                        "loss": train_avg_loss,
                        # "mean_abs": train_avg_loss[1],
                    },
                }
            )
            eval_end_time = time.time()
            eval_wall = eval_end_time - eval_start_time

        save_model_state = (valid_avg_loss < loss_valid_best) and only_best
        # print(f"save_model_state {save_model_state} ({valid_avg_loss[0] } < {loss_valid_best} and {only_best})")

        if step == checkpoint or save_model_state:
            if save_model_state:
                best_model_state = copy.deepcopy(model.state_dict())
                loss_valid_best = copy.deepcopy(valid_avg_loss)
            if step == checkpoint:
                checkpoint = next(checkpoint_generator)
                assert checkpoint > step

            results = {
                "history": history,
                "state_best": best_model_state,
                "state_last": copy.deepcopy(model.state_dict()),
            }

            msg = f"Iteration {step + 1:4d}   "
            msg += f"train loss = {train_avg_loss:8.4f}   "
            msg += f"valid loss = {valid_avg_loss:8.4f}   "
            msg += f"elapsed time = {time.strftime('%H:%M:%S', time.gmtime(wall))} "
            if evaluate_model:
                msg += f"(eval time = {time.strftime('%H:%M:%S', time.gmtime(eval_wall))})  "
            if save_model_state:
                msg += "> state saved"

            print(msg)

            with open(str(run_name) + ".torch", "wb") as f:
                torch.save(results, f)

        if scheduler is not None:
            scheduler.step()


def evaluate_classifier(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    threshold: float,
    device: str,
) -> tuple[float, float]:
    """
    Evaluates the classifier model on the given dataloader.

    Parameters:
        model (torch.nn.Module): The neural network model to be evaluated.
        dataloader (DataLoader): The DataLoader containing the data to evaluate the model on.
        loss_fn (torch.nn.Module): The loss function used for evaluation.
        threshold (torch.nn.Module): probability threshold for binary accuracy.
        device (str): The device (e.g., 'cuda' or 'cpu') on which to perform the evaluation.

    Returns:
        Tuple[float, float, float]: The average loss, average binary auroc and
        average binary accuracy across all batches in the dataloader.
    """
    model.eval()
    loss_cumulative = 0.0
    metric_accuracy_cumulative = 0.0
    metric_auroc_cumulative = 0.0
    # start_time = time.time()
    with torch.no_grad():
        for j, d in enumerate(dataloader):
            d.to(device)
            output = model(d)
            loss = loss_fn(output, d.target).cpu()
            accuracy = binary_accuracy(
                output.reshape(-1), d.target.reshape(-1), threshold=threshold
            ).cpu()
            auroc = binary_auroc(output.reshape(-1), d.target.reshape(-1)).cpu()
            loss_cumulative = loss_cumulative + loss.detach().item()
            metric_accuracy_cumulative = metric_accuracy_cumulative + accuracy
            metric_auroc_cumulative = metric_auroc_cumulative + auroc
    return (
        loss_cumulative / len(dataloader),
        metric_accuracy_cumulative / len(dataloader),
        metric_auroc_cumulative / len(dataloader),
    )


def train_classifier(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader_train: DataLoader,
    dataloader_valid: DataLoader,
    loss_fn: torch.nn.Module,
    threshold: float,
    run_name: str,
    max_iter: int = 101,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    only_best: bool = True,
    device: str = "cpu",
) -> None:
    """
    Trains the model using the given dataloaders, optimizer, and loss functions.

    Parameters:
        model (torch.nn.Module): The neural network model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        dataloader_train (DataLoader): The DataLoader for training data.
        dataloader_valid (DataLoader): The DataLoader for validation data.
        loss_fn (torch.nn.Module): The loss function used for training.
        threshold (torch.nn.Module): probability threshold for binary accuracy.
        run_name (str): The name of the run, used for saving model checkpoints.
        max_iter (int): Maximum number of training iterations.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler, optional.
        only_best (bool): Flag to save only the best model based on validation loss.
        device (str): The device (e.g., 'cuda' or 'cpu') on which to train the model.

    Saves:
        Model checkpoints and training history to a file named "{run_name}.torch".
    """
    model.to(device)

    checkpoint_generator = loglinspace(0.3, 5)
    checkpoint = next(checkpoint_generator)
    start_time = time.time()

    try:
        model.load_state_dict(torch.load(run_name + ".torch")["state_best_accuracy"])
        best_model_state_accu = copy.deepcopy(model.state_dict())
        model.load_state_dict(torch.load(run_name + ".torch")["state_best_loss"])
        best_model_state_loss = copy.deepcopy(model.state_dict())
    except:  # noqa
        results = {}
        history = []
        s0 = 0
    else:
        results = torch.load(run_name + ".torch")
        history = results["history"]
        s0 = history[-1]["step"] + 1
    try:
        history[-1]["valid"]
    except (KeyError, IndexError):
        loss_valid_best = 9.9999e99
        accuracy_valid_best = -1
    else:
        loss_valid_best = min([d["valid"]["loss"] for d in history])
        accuracy_valid_best = max([d["valid"]["accuracy"] for d in history])

    for step in range(max_iter):
        model.train()
        loss_cumulative = 0.0
        metric_accuracy_cumulative = 0.0
        metric_auroc_cumulative = 0.0

        for j, d in tqdm(
            enumerate(dataloader_train), total=len(dataloader_train), bar_format=BAR_FROMAT
        ):
            d.to(device)
            output = model(d)
            loss = loss_fn(output, d.target).cpu()
            accuracy = binary_accuracy(
                output.reshape(-1), d.target.reshape(-1), threshold=threshold
            ).cpu()
            auroc = binary_auroc(output.reshape(-1), d.target.reshape(-1)).cpu()
            loss_cumulative = loss_cumulative + loss.detach().item()
            metric_accuracy_cumulative = metric_accuracy_cumulative + accuracy
            metric_auroc_cumulative = metric_auroc_cumulative + auroc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end_time = time.time()
        wall = end_time - start_time

        # save history

        evaluate_model = (step == checkpoint) or only_best
        # print(f"evaluate_model {evaluate_model} ({step} == {checkpoint} or {only_best})")
        if evaluate_model:
            eval_start_time = time.time()
            valid_avg_loss = evaluate_classifier(
                model, dataloader_valid, loss_fn, threshold, device
            )
            # train_avg_loss = evaluate(model, dataloader_train, loss_fn, loss_fn_mae, device)
            train_avg_loss = (
                loss_cumulative / len(dataloader_train),
                metric_accuracy_cumulative / len(dataloader_train),
                metric_auroc_cumulative / len(dataloader_train),
            )

            history.append(
                {
                    "step": s0 + step,
                    "wall": wall,
                    "batch": {
                        "loss": loss.item(),
                        "accuracy": accuracy,
                        "auroc": auroc,
                    },
                    "valid": {
                        "loss": valid_avg_loss[0],
                        "accuracy": valid_avg_loss[1],
                        "auroc": valid_avg_loss[2],
                    },
                    "train": {
                        "loss": train_avg_loss[0],
                        "accuracy": train_avg_loss[1],
                        "auroc": train_avg_loss[2],
                    },
                }
            )
            eval_end_time = time.time()
            eval_wall = eval_end_time - eval_start_time

        save_model_state_loss = (valid_avg_loss[0] < loss_valid_best) and only_best
        save_model_state_accu = (valid_avg_loss[1] > accuracy_valid_best) and only_best

        # print(f"save_model_state {save_model_state} ({valid_avg_loss[0] } < {loss_valid_best} and {only_best})")

        if step == checkpoint or save_model_state_loss or save_model_state_accu:
            if save_model_state_loss:
                best_model_state_loss = copy.deepcopy(model.state_dict())
                loss_valid_best = copy.deepcopy(valid_avg_loss[0])
            if save_model_state_accu:
                best_model_state_accu = copy.deepcopy(model.state_dict())
                accuracy_valid_best = copy.deepcopy(valid_avg_loss[1])
            if step == checkpoint:
                checkpoint = next(checkpoint_generator)
                assert checkpoint > step

            results = {
                "history": history,
                "state_best_loss": best_model_state_loss,
                "state_best_accuracy": best_model_state_accu,
                "state_last": copy.deepcopy(model.state_dict()),
            }

            msg = f"Iteration {step + 1:4d}   "
            msg += f"train loss = {train_avg_loss[0]:8.4f}, "
            msg += f"valid loss = {valid_avg_loss[0]:8.4f} | "
            msg += f"train accuracy = {train_avg_loss[1]:7.4f}, "
            msg += f"valid accuracy = {valid_avg_loss[1]:7.4f} | "
            msg += f"train auroc = {train_avg_loss[2]:7.4f}, "
            msg += f"valid auroc = {valid_avg_loss[2]:7.4f}    "
            msg += f"elapsed time = {time.strftime('%H:%M:%S', time.gmtime(wall))} "
            if evaluate_model:
                msg += f"(eval time = {time.strftime('%H:%M:%S', time.gmtime(eval_wall))})  "
            if save_model_state_loss:
                msg += "> best state saved (loss)"
            if save_model_state_accu:
                msg += "> best state saved (accuracy)"

            print(msg)

            with open(run_name + ".torch", "wb") as f:
                torch.save(results, f)

        if scheduler is not None:
            scheduler.step()
