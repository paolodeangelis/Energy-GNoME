"""source: https://github.com/ninarina12/phononDoS_tutorial ?
"""

from typing import Union

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import Gate
from e3nn.nn.models.gate_points_2101 import Convolution, smooth_cutoff, tp_path_exists
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius_graph
import torch_geometric as tg
from torch_geometric.data import Data
import torch_scatter


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


class PeriodicNetworkClassifier(Network):
    """
    An extension of the Network class to handle periodic systems for binary-classification.

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

        # Modify the final layer to output a single value for binary classification
        self.final_layer = nn.Linear(self.irreps_out.dim, 1)

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

        # Apply the final layer and sigmoid activation
        output = torch.sigmoid(self.final_layer(output))

        # if pool_nodes was set to True, use scatter_mean to aggregate
        if self.pool is True:
            output = torch_scatter.scatter_mean(
                output, data.batch, dim=0
            )  # take mean over atoms per example

        # maxima, _ = torch.max(output, dim=1)
        # output = output.div(maxima.unsqueeze(1))

        return output
