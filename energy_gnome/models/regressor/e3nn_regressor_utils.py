"""source: https://github.com/ninarina12/phononDoS_tutorial ?
"""

from collections.abc import Generator
import copy
import math
import time
from typing import Optional, Union

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import Gate
from e3nn.nn.models.gate_points_2101 import Convolution, smooth_cutoff, tp_path_exists
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_cluster import radius_graph
import torch_geometric as tg
from torch_geometric.data import Data
import torch_scatter
from torcheval.metrics.functional import binary_accuracy, binary_auroc
from tqdm.auto import tqdm

BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"


class CustomCompose(torch.nn.Module):
    """
    A custom composition module to sequentially apply two different modules.

    This module takes two modules as input and applies them sequentially. It stores the
    intermediate output and the final output for further use.

    Attributes:
        first (torch.nn.Module): The first module to be applied.
        second (torch.nn.Module): The second module to be applied after the first.
        irreps_in (e3nn.o3.Irreps): The input irreducible representations of the first module.
        irreps_out (e3nn.o3.Irreps): The output irreducible representations of the second module.
    """

    def __init__(self, first: torch.nn.Module, second: torch.nn.Module) -> None:
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        self.first_out = x.clone()
        x = self.second(x)
        self.second_out = x.clone()
        return x


class Network(torch.nn.Module):
    """
    An equivariant neural network using the e3nn framework.

    This network is designed to handle graph-like data where nodes and edges have associated features.
    It can process features that are equivariant under the O(3) group, making it suitable for tasks
    like molecular property prediction or other 3D structure-related problems.

    Parameters:
        irreps_in (e3nn.o3.Irreps or None):
            The representation of the input features. Set to `None` if nodes do not have input features.
        irreps_hidden (e3nn.o3.Irreps):
            The representation of the hidden features.
        irreps_out (e3nn.o3.Irreps):
            The representation of the output features.
        irreps_node_attr (e3nn.o3.Irreps or None):
            The representation of the node attributes. Set to `None` if nodes do not have attributes.
        irreps_edge_attr (e3nn.o3.Irreps):
            The representation of the edge attributes. Edge attributes are described by :math:`h(r) Y(\vec r / r)`,
            where :math:`h` is a smooth function that vanishes at `max_radius`, and :math:`Y`
            are the spherical harmonics polynomials.
        layers (int):
            The number of gates (non-linearities) in the network.
        max_radius (float):
            The maximum radius for the convolution.
        number_of_basis (int):
            The number of bases on which the edge lengths are projected.
        radial_layers (int):
            The number of hidden layers in the radial fully connected network.
        radial_neurons (int):
            The number of neurons in the hidden layers of the radial fully connected network.
        num_neighbors (float):
            The typical number of nodes within a distance of `max_radius`.
        num_nodes (float):
            The typical number of nodes in a graph.

    The network utilizes spherical harmonics and radial functions to represent 3D geometric information
    in a manner that is equivariant under rotations and translations.
    """

    def __init__(
        self,
        irreps_in: str | None,
        irreps_out: str,
        irreps_node_attr: str | None,
        layers: int,
        mul: int,
        lmax: int,
        max_radius: float,
        number_of_basis: int = 10,
        radial_layers: int = 1,
        radial_neurons: int = 100,
        num_neighbors: float = 1.0,
        num_nodes: float = 1.0,
        reduce_output: bool = True,
    ) -> None:
        super().__init__()
        self.mul = mul
        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.reduce_output = reduce_output

        self.irreps_in = o3.Irreps(irreps_in) if irreps_in is not None else None
        self.irreps_hidden = o3.Irreps(
            [(self.mul, (l_, p_)) for l_ in range(lmax + 1) for p_ in [-1, 1]]
        )
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attr = (
            o3.Irreps(irreps_node_attr) if irreps_node_attr is not None else o3.Irreps("0e")
        )
        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)

        self.input_has_node_in = irreps_in is not None
        self.input_has_node_attr = irreps_node_attr is not None

        irreps = self.irreps_in if self.irreps_in is not None else o3.Irreps("0e")

        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        self.layers = torch.nn.ModuleList()

        for _ in range(layers):
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_hidden
                    if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)
                ]
            )
            irreps_gated = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_hidden
                    if ir.l > 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)
                ]
            )
            ir = "0e" if tp_path_exists(irreps, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(
                irreps_scalars,
                [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )
            conv = Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                gate.irreps_in,
                number_of_basis,
                radial_layers,
                radial_neurons,
                num_neighbors,
            )
            irreps = gate.irreps_out
            self.layers.append(CustomCompose(conv, gate))

        self.layers.append(
            Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                self.irreps_out,
                number_of_basis,
                radial_layers,
                radial_neurons,
                num_neighbors,
            )
        )

    def preprocess(self, data: Data | dict[str, torch.Tensor]) -> torch.Tensor:
        if "batch" in data:
            batch = data["batch"]
        else:
            batch = data["pos"].new_zeros(data["pos"].shape[0], dtype=torch.long)

        if "edge_index" in data:
            edge_src = data["edge_index"][0]  # edge source
            edge_dst = data["edge_index"][1]  # edge destination
            edge_vec = data["edge_vec"]

        else:
            edge_index = radius_graph(data["pos"], self.max_radius, batch)
            edge_src = edge_index[0]
            edge_dst = edge_index[1]
            edge_vec = data["pos"][edge_src] - data["pos"][edge_dst]

        # dos = data['dos']
        # dos_energy = data['dos_energy']
        # dos_fermi = data['dos_fermi']

        return batch, edge_src, edge_dst, edge_vec  # , dos, dos_energy

    def forward(self, data: Data | dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Evaluates the network on the provided data.

        This method processes the input data through the network. It computes spherical harmonics and
        edge attributes, and applies the network layers to the input features.

        Parameters:
            data (Union[Data, dict[str, torch.Tensor]]): The input data to the network. This can be a
            `torch_geometric.data.Data` object or a dictionary. The expected contents include:
                - `pos`: The positions of the nodes (atoms).
                - `x`: The input features of the nodes, optional.
                - `z`: The attributes of the nodes, such as atom type, optional.
                - `batch`: Information about the graph to which each node belongs, optional.

        Returns:
            torch.Tensor: The output of the network after processing the input data. The output tensor
            dimensions depend on the network configuration and whether `reduce_output` is set to True.
        """
        batch, edge_src, edge_dst, edge_vec = self.preprocess(
            data
        )  # , dos, dos_energy = self.preprocess(data)
        edge_sh = o3.spherical_harmonics(
            self.irreps_edge_attr, edge_vec, True, normalization="component"
        )
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis="gaussian",
            cutoff=False,
        ).mul(self.number_of_basis**0.5)
        edge_attr = smooth_cutoff(edge_length / self.max_radius)[:, None] * edge_sh

        if self.input_has_node_in and "x" in data:
            assert self.irreps_in is not None
            x = data["x"]
        else:
            assert self.irreps_in is None
            x = data["pos"].new_ones((data["pos"].shape[0], 1))

        if self.input_has_node_attr and "z" in data:
            z = data["z"]
        else:
            assert self.irreps_node_attr == o3.Irreps("0e")
            z = data["pos"].new_ones((data["pos"].shape[0], 1))

        for lay in self.layers:
            x = lay(
                x, z, edge_src, edge_dst, edge_attr, edge_length_embedded
            )  # , dos, dos_energy)

        # print(x, x.shape)

        if self.reduce_output:
            return torch.scatter(x, batch, dim=0).div(self.num_nodes**0.5)
        else:
            return x


class PeriodicNetwork(Network):
    """
    An extension of the Network class to handle periodic systems.

    This class extends the equivariant network to handle periodic systems by incorporating
    additional features such as mass-weighted one-hot encoding.

    Attributes:
        pool (bool): If True, pooling over atom contributions is performed.
        em (nn.Linear): A linear layer to embed mass-weighted one-hot encoding.
    """

    def __init__(self, in_dim: int, em_dim: int, **kwargs) -> None:
        # override the `reduce_output` keyword to instead perform an averge over atom contributions
        self.pool = False
        if kwargs["reduce_output"] is True:
            kwargs["reduce_output"] = False
            self.pool = True

        super().__init__(**kwargs)

        # embed the mass-weighted one-hot encoding
        self.em = nn.Linear(in_dim, em_dim)
        print(self.em)

    def forward(self, data: tg.data.Data | dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the PeriodicNetwork.

        Parameters:
            data (Union[tg.data.Data, dict[str, torch.Tensor]]): Input data containing node and edge features.

        Returns:
            torch.Tensor: The output of the network after processing the input data.
        """
        data.x = F.relu(self.em(data.x))
        # print(data.x.shape)
        data.z = F.relu(self.em(data.z))
        output = super().forward(data)
        output = torch.relu(output)

        # if pool_nodes was set to True, use scatter_mean to aggregate
        if self.pool is True:
            output = torch_scatter.scatter_mean(
                output, data.batch, dim=0
            )  # take mean over atoms per example

        # maxima, _ = torch.max(output, dim=1)
        # output = output.div(maxima.unsqueeze(1))

        return output


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
            enumerate(dataloader_train), total=len(dataloader_train), bar_format=BAR_FORMAT
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
