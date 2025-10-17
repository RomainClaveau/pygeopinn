#!/usr/bin/python3
# -*- coding: utf-8 -*-

# This file is part of pygeopinn
# (c) 2025 - Geodynamo (ISTerre) (GNU GPLv3)
# Developed by Romain Claveau (https://orcid.org/0000-0001-5961-0370)

__version__ = 0

# Imports
from torch import nn, autograd, optim, amp
from torch import tensor, Tensor, no_grad, ones_like, concat, autocast, bfloat16, \
    save, load, float32, sin, cos, manual_seed, use_deterministic_algorithms
from numpy import ndarray, min, max, meshgrid
from numpy import pi as π
from tqdm import tqdm

manual_seed(0)
use_deterministic_algorithms(True)

class pygeopinn:
    r"""
    `pygeopinn` is a library to ease the use of neural networks in order to solve
    the inverse geodynamo problem. The problem is solved in the physical domain.
    Along with the azimuthal and polar components of the flow, the magnetic field
    is co-estimated.
    """    
    class __network__(nn.Module):
        """
        Nested class for creating a single neural network.
        """
        class Sine(nn.Module):
            """
            Creating the Sine activation function
            """
            def __init__(self):
                super().__init__()
            def forward(self, x):
                return sin(x)
            
        def __init__(self, nb_layers: int, nb_neurons: int) -> None:
            """
            Initializing the network.
            - (int) `nb_layers` the number of hidden layers.
            - (int) `nb_neurons` the number of neurons.
            """
            super().__init__()

            # Initializing the layers
            layers = []

            # Creating the first layer
            layers.append(nn.Linear(2, nb_neurons, dtype=float32))
            layers.append(self.Sine())

            # Creating the hidden layers
            for _ in range(nb_layers):
                layers.append(nn.Linear(nb_neurons, nb_neurons, dtype=float32))
                layers.append(self.Sine())

            # Creating the last layer
            layers.append(nn.Linear(nb_neurons, 3, dtype=float32))

            # Creating the network
            self.net = nn.Sequential(*layers)

            # Initializing the weights
            with no_grad():
                for module in self.net.modules():
                    if module._get_name() == "Linear":
                        nn.init.xavier_uniform_(module.weight)

        def forward(self, input) -> Tensor:
            """
            Forward method.
            - (tensor) `input` the input tensor.
            """
            return self.net(input)

    def __init__(self, nb_layers: int = 5, nb_neurons: int = 32, verbose: bool = True) -> None:
        r"""
        Initializing the library, and the network.
        - (int) `nb_layers` the number of hidden layers (default: 5)
        - (int) `nb_neurons` the number of neurons (default: 32)
        - (bool) `verbose` enable the verbose mode (default: true)
        """
        self.network = self.__network__(nb_layers, nb_neurons)

        self.nb_layers = nb_layers
        self.nb_neurons = nb_neurons
        self.verbose = verbose

        self.observations = {}
        self.tensors = {}
        self.losses = {}

        if self.verbose:
            print("The library was successfully initialized.")

    def set_grid(self, thetas: ndarray, phis: ndarray) -> None:
        """
        Setting the grid.
        - (array) `thetas` the grid along the θ axis
        - (array) `phis` the grid along the φ axis
        """
        if min(thetas) < 0 or max(thetas) > π:
            raise Exception("The grid along the θ-axis must be between 0 and π radians.")
        
        if min(phis) < 0 or max(phis) > 2 * π:
            raise Exception("The grid along the φ-axis must be between 0 and 2π radians.")
        
        self.grid = {"thetas": thetas, "phis": phis}

        if self.verbose:
            print("The grid was successfully set.")

    def set_observation(self, name: str, observation: ndarray, overwrite: bool = True) -> None:
        """
        Adding an observation.
        - (str) `name` the name of the observation.
        - (array) `obsbervation` the 2D map of the observation.
        - (bool) `overwrite` allows overwriting an observation.
        """
        if name in self.observations and not overwrite:
            raise Exception(f"The observation {name} already exists.")
        
        if not isinstance(observation, (ndarray)):
            raise Exception("The observation must be an array.")
        
        if len(observation.shape) != 2:
            raise Exception("The observation must have 2 dimensions.")
        
        self.observations.update({name: observation})

        if self.verbose:
            print(f"The observation {name} was successfully added.")

    def _array_to_tensor(self, array: ndarray, requires_grad: bool = True) -> Tensor:
        """
        Converting a numpy array into a torch tensor.
        - (array) `array` the array to convert.
        - (bool) `requires_grad` flag for autograd to record, or not, operations.
        """
        return tensor(array, requires_grad=requires_grad, dtype=float32)
    
    def _autograd(self, tensor: Tensor, inputs: list, retain_graph: bool = True, create_graph: bool = True) -> tuple:
        """
        Computing the gradients of the tensor wrt the inputs.
        - (tensor) `tensor` the tensor from which the gradients are computed.
        - (list) `inputs` the list of inputs.
        - (bool) `retain_graph` keeping in memory the gradient graph.
        - (bool) `create_graph` creating the graph to compute higher-order derivatives.
        """
        return autograd.grad(tensor, inputs, ones_like(tensor), retain_graph, create_graph)
    
    def initialize(self) -> None:
        """
        Initializing the observations before the training.
        """
        for name in self.observations.keys():
            self.tensors.update({
                name: self._array_to_tensor(self.observations[name].reshape(-1, 1))
            })

        thetas_grid, phis_grid = meshgrid(self.grid["thetas"], self.grid["phis"], indexing="ij")

        # Useful for reshaping torch's squeezed fields later
        self.shape = thetas_grid.shape

        self.tensors.update({
            "thetas": self._array_to_tensor(thetas_grid.reshape(-1, 1)),
            "phis": self._array_to_tensor(phis_grid.reshape(-1, 1))
        })

        self.inputs = concat([self.tensors["thetas"], self.tensors["phis"]], dim=1)

        if self.verbose:
            print("Everything is ready for the training.")

    def train(self, nb_epochs: int = 10000, init: bool = True):
        """
        Training the networks.
        - (int) `nb_epochs` the number of epochs.
        - (bool) `init` initializing before starting the training
        """
        self.network.train()

        if init:
            self.lower_loss = float('inf')
            self.initialize()

        # TODO: Check if everything is ready before starting

        optimizer = optim.AdamW(self.network.parameters())
        scaler = amp.GradScaler("cpu")

        tqdm_format = "{percentage:3.2f}% ({remaining} remaining) | Loss = {postfix[0]:.3E}"
        
        with tqdm(total=nb_epochs, bar_format=tqdm_format, postfix=[self.lower_loss]) as pg:
            for epoch in range(nb_epochs):
                optimizer.zero_grad(set_to_none=True)

                # Mixed precision to speed up calculations
                with autocast(device_type="cpu", dtype=bfloat16, enabled=True):
                    loss = self.loss(self.inputs)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                nn.utils.clip_grad_norm_(self.network.parameters(), 1)

                scaler.step(optimizer)
                scaler.update()

                with no_grad():
                    if loss.item() < self.lower_loss:
                        save(self.network.state_dict(), "best_model.pt")

                        self.lower_loss = loss.item()
                        pg.postfix[0] = self.lower_loss

                pg.update()

        self.network.load_state_dict(load("best_model.pt"))

    def loss(self, inputs: Tensor, evaluate: bool = False) -> Tensor | dict:
        r"""
        Computing the loss function.
        - (tensor) `inputs` the inputs form which the loss is estimated.
        - (bool) `evaluate` returning the fields instead of the loss.
        """
        rC = 3485

        # Retrieving the predictions
        predictions = self.network(inputs)
        t = predictions[...,0:1]
        s = predictions[...,1:2]
        br = 1e6 * predictions[...,2:3]

        # Retrieving observations
        br_obs = self.tensors["br"]
        dbrdt_obs = self.tensors["dbrdt"]

        θ = self.tensors["thetas"]
        φ = self.tensors["phis"]
        sinθ = sin(θ).clamp(1e-1, 1)
        cosθ = cos(θ)

        dtdθ, dtdφ = self._autograd(t, [θ, φ])
        dsdθ, dsdφ = self._autograd(s, [θ, φ])

        # Retrieving uθ and uφ
        uθ = (1 / sinθ) * dtdφ + dsdθ
        uφ = -(1 / sinθ) * dtdθ + dsdφ

        # Computing derivatives
        duθdθ, duθdφ = self._autograd(uθ, [θ,φ])
        duφdθ, duφdφ = self._autograd(uφ, [θ,φ])
        dbrdθ, dbrdφ = self._autograd(br, [θ,φ])

        # Computing divergence and gradient operators
        divh_uh = (1 / (rC * sinθ)) * (duθdθ * sinθ + uθ * cosθ + duφdφ)
        gradθ_br = (1 / rC) * dbrdθ
        gradφ_br = (1 / (rC * sinθ)) * dbrdφ

        dbrdt = -(br * divh_uh + gradθ_br * uθ + gradφ_br * uφ)

        loss = (dbrdt_obs - dbrdt).pow(2).mean() / dbrdt_obs.pow(2).mean()
        loss += (br_obs - br).pow(2).mean() / br_obs.pow(2).mean()

        if evaluate:
            self.predictions = {
                "br": br.detach().numpy().reshape(self.shape), "t": t.detach().numpy().reshape(self.shape),
                "s": s.detach().numpy().reshape(self.shape), "uθ": uθ.detach().numpy().reshape(self.shape),
                "uφ": uφ.detach().numpy().reshape(self.shape), "dbrdt": dbrdt.detach().numpy().reshape(self.shape)
            }
            return self.predictions

        return loss
    
    def evaluate(self) -> dict:
        """
        Evaluating the networks predictions.
        """
        self.network.load_state_dict(load("best_model.pt"))
        self.network.eval()

        return self.loss(self.inputs, True)