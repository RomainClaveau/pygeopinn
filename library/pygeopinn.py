#!/usr/bin/python3
# -*- coding: utf-8 -*-

# This file is part of pygeopinn
# (c) 2025 - Geodynamo (ISTerre) (GNU GPLv3)
# Developed by Romain Claveau (https://orcid.org/0000-0001-5961-0370)

__version__ = 0

# Imports
import torch
import numpy
import scipy
from numpy import pi as π
from tqdm import tqdm

rE = 6371.2
rC = 3485.0

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

class pygeopinn:
    r"""
    `pygeopinn` is a library to ease the use of neural networks in order to solve
    the inverse geodynamo problem. The problem is solved in the physical domain.
    Along with the azimuthal and polar components of the flow, the magnetic field
    is co-estimated.
    """    
    class _network(torch.nn.Module):
        """
        Nested class for creating a single neural network.
        """
        class Sine(torch.nn.Module):
            """
            Creating the Sine activation function
            """
            def __init__(self, w0=1.0):
                super().__init__()
                self.w0 = w0
            def forward(self, x):
                return torch.sin(self.w0 * x)
            
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
            layers.append(torch.nn.Linear(3, nb_neurons, dtype=torch.float32))
            layers.append(self.Sine(30.0))

            # Creating the hidden layers
            for _ in range(nb_layers):
                layers.append(torch.nn.Linear(nb_neurons, nb_neurons, dtype=torch.float32))
                layers.append(self.Sine(1.0))

            # Creating the last layer
            layers.append(torch.nn.Linear(nb_neurons, 3, dtype=torch.float32))

            # Creating the network
            self.net = torch.nn.Sequential(*layers)

            n = 1

            # Initializing the weights
            with torch.no_grad():
                for m in self.net:
                    if isinstance(m, torch.nn.Linear):
                        w0 = 30.0 if n == 1 else 1.0
                        bound = numpy.sqrt(6 / m.in_features) / w0
                        torch.nn.init.uniform_(m.weight, -bound, bound)
                        torch.nn.init.zeros_(m.bias)
                        n += 1

        def forward(self, input) -> torch.Tensor:
            """
            Forward method.
            - (tensor) `input` the input tensor.
            """
            return self.net(input)
        
    class _spectral:
        """
        Module for computing the spherical harmonics and forward / backward transforms.
        """
        def __init__(self, thetas: numpy.ndarray, phis: numpy.ndarray, lmax: int = 30) -> None:
            """
            Initializing the spectral module
            - (array) `thetas` the grid along the θ-axis.
            - (array) `phis` the grid along the φ-axis.
            - (lmax) `lmax` the truncation degree.
            """
            if numpy.min(thetas) < 0 or numpy.max(thetas) > π:
                raise Exception("The grid along the θ-axis must be between 0 and π radians.")
            
            if numpy.min(phis) < 0 or numpy.max(phis) > 2 * π:
                raise Exception("The grid along the φ-axis must be between 0 and 2π radians.")
            
            if lmax < 1 or lmax > 100:
                raise Exception("The truncation degree must be between 1 and 100.")
            
            self.thetas = torch.tensor(thetas, dtype=torch.float64)
            self.phis = torch.tensor(phis, dtype=torch.float64)
            self.lmax = lmax

            self._ycos, self._ysin = self._spherical_harmonics(self.lmax)

        def _spherical_harmonics(self, lmax: int = 30) -> tuple:
            """
            Computing the spherical harmonics up to the truncation degree lmax
            on the specified grid.
            - (int) `lmax` the truncation degree
            """
            if lmax < 1 or lmax > 100:
                raise Exception("The truncation degree must be between 1 and 100.")
            
            if lmax > self.lmax:
                self.lmax = lmax

            l = torch.arange(0, self.lmax + 1, 1)
            L, M = torch.meshgrid(l, l, indexing="ij")

            θ, φ = torch.meshgrid(self.thetas, self.phis, indexing="ij")
            cosθ = torch.cos(θ)

            def δ(x):
                return (x == 0).to(torch.float32)
            
            def Γ(x):
                return torch.tensor(scipy.special.factorial(x), dtype=torch.float64)
            
            schmidt = (-1)**M * torch.sqrt(torch.abs((2 - δ(M)) * Γ(L - M) / Γ(L + M)))

            p, *_ = scipy.special.assoc_legendre_p_all(lmax, lmax, cosθ)
            p = torch.tensor(p[:,:(lmax+1),...], dtype=torch.float64)
            p = torch.einsum("ijkl,ij->ijkl", p, schmidt)

            mφ = torch.einsum("ij,kl->ijkl", M, φ)
            cosmφ = torch.cos(mφ)
            sinmφ = torch.sin(mφ)

            print(f"Computed for lmax = {lmax}")

            return cosmφ * p, sinmφ * p
        
        def ycos(self, lmax: int) -> torch.Tensor:
            """
            Retrieving the cosine part of the spherical harmonics.
            (int) `lmax` the required truncation degree.
            """
            if lmax < 1 or lmax > 100:
                raise Exception("The truncation degree must be between 1 and 100.")
            
            if lmax > self.lmax:
                self._ycos, self._ysin = self._spherical_harmonics(lmax)

            return self._ycos[:lmax+1,:lmax+1,...]
        
        def ysin(self, lmax: int) -> torch.Tensor:
            """
            Retrieving the sine part of the spherical harmonics.
            (int) `lmax` the required truncation degree.
            """
            if lmax < 1 or lmax > 100:
                raise Exception("The truncation degree must be between 1 and 100.")
            
            if lmax > self.lmax:
                self._ycos, self._ysin = self._spherical_harmonics(lmax)

            return self._ysin[:lmax+1,:lmax+1,...]
        
        def forward(self, x: torch.Tensor, lmax: int) -> tuple:
            """
            Performing forward spectral transformation of a field.
            - (tensor) `x` the field to process.
            - (int) `lmax` the truncation degree.
            """
            if lmax < 1 or lmax > 100:
                raise Exception("The truncation degree must be between 1 and 100.")
            
            ycos = self.ycos(lmax)
            ysin = self.ysin(lmax)

            if ycos.shape[-2:] != x.shape[-2:]:
                raise Exception("The provided tensor does not match with the expected spatial grid.")

            l = torch.arange(0, lmax + 1, 1)
            L, M = torch.meshgrid(l, l, indexing="ij")

            θ, φ = torch.meshgrid(self.thetas, self.phis, indexing="ij")
            sinθ = torch.sin(θ)
            dθ = torch.gradient(self.thetas)[0].mean()
            dφ = torch.gradient(self.phis)[0].mean()
            dΩ = sinθ * dθ * dφ

            weight = (2 * L + 1) / (4 * torch.pi)

            if len(x.shape) == 3:
                coeffs_cos = weight * torch.einsum("ijkl,mkl->mij", ycos, x * dΩ)
                coeffs_sin = weight * torch.einsum("ijkl,mkl->mij", ysin, x * dΩ)
            else:
                coeffs_cos = weight * torch.einsum("ijkl,kl->ij", ycos, x * dΩ)
                coeffs_sin = weight * torch.einsum("ijkl,kl->ij", ysin, x * dΩ)

            return coeffs_cos, coeffs_sin
        
        def backward(self, xcos: torch.Tensor, xsin: torch.Tensor, lmax: int) -> torch.Tensor:
            """
            Performing backward spectral transformation of a field.
            - (tensor) `xcos` the cosine part of the spectral coefficients.  
            - (tensor) `xsin` the sine part of the spectral coefficients.  
            - (int) `lmax` the truncation degree.  
            """
            if lmax < 1 or lmax > 100:
                raise Exception("The truncation degree must be between 1 and 100.")
            
            if lmax > xcos.shape[-1] - 1:
                raise Exception("Spectral coefficients have lower truncation degree than requested.")
            
            if len(xcos.shape) < 2 or len(xsin.shape) < 2:
                raise Exception("Spectral coeffients must be at least 2 dimensional arrays.")
            
            ycos = self.ycos(lmax)
            ysin = self.ysin(lmax)

            return torch.tensordot(xcos[...,:lmax+1,:lmax+1], ycos, dims=2) + \
                torch.tensordot(xsin[...,:lmax+1,:lmax+1], ysin, dims=2)
        
        def spectrum(self, xs: list, lmax: int, output: str) -> None | torch.Tensor:
            """
            Compute the spectrum associated with the field.
            - (list) `xs` the tensors from which the spectrum is computed.
            - (int) `lmax` the truncation degree.
            - (str) `output` which spectrum must be computed (`spectrum_br`,
                `spectrum_dbrdt`)
            """
            if not isinstance(xs, list):
                raise Exception("The tensor(s) must be given through a list.")
            
            coefficients = []

            for x in xs:
                coefficients.append(self.forward(x, lmax))

            l = torch.arange(0, lmax + 1, 1)
            L, _ = torch.meshgrid(l, l, indexing="ij")

            # Spectral
            if output in ["spectrum_dbrdt", "spectrum_br"]:
                xcos, xsin = coefficients[0]
                xcos /= (L + 1) * (rE / rC)**(L + 2)
                xsin /= (L + 1) * (rE / rC)**(L + 2)
                return (l + 1) * (xcos.pow(2) + xsin.pow(2)).sum(dim=2)

            if output == "spectrum_u":
                t_cos, t_sin = coefficients[0]
                s_cos, s_sin = coefficients[1]
                return (l * (l + 1) / (2 * l + 1)) * (t_cos.pow(2) + t_sin.pow(2) + \
                    s_cos.pow(2) + s_sin.pow(2)).sum(dim=2)
            
            # Physical
            if output in ["squared_norm_br", "squared_norm_dbrdt"]:
                xcos, xsin = coefficients[0]
                xcos /= (L + 1) * (rE / rC)**(L + 2)
                xsin /= (L + 1) * (rE / rC)**(L + 2)
                S = (l + 1) * (xcos.pow(2) + xsin.pow(2)).sum(dim=2)
                return (((l + 1) / (2 * l + 1)) * (rE / rC)**(2 * l + 4) * S).sum(dim=1)
            
            if output == "squared_norm_u":
                t_cos, t_sin = coefficients[0]
                s_cos, s_sin = coefficients[1]
                S = (l * (l + 1) / (2 * l + 1)) * (t_cos.pow(2) + t_sin.pow(2) + \
                    s_cos.pow(2) + s_sin.pow(2)).sum(dim=2)
                return S.sum(dim=1)
            
    def __init__(self, nb_layers: int = 5, nb_neurons: int = 32, verbose: bool = True) -> None:
        r"""
        Initializing the library, and the network.
        - (int) `nb_layers` the number of hidden layers (default: 5)
        - (int) `nb_neurons` the number of neurons (default: 32)
        - (bool) `verbose` enable the verbose mode (default: true)
        """
        self.network = self._network(nb_layers, nb_neurons)

        self.nb_layers = nb_layers
        self.nb_neurons = nb_neurons
        self.verbose = verbose

        self.observations = {}
        self.tensors = {}
        self.losses = {}
        self.history = {}
        self.options = {}

        # Temporary
        self.P_inv_zz = self._array_to_tensor(numpy.loadtxt("misc/P_inv_10deg.mat"))
        self.z_mean = self._array_to_tensor(numpy.loadtxt("misc/z_10deg.vec"))

        if self.verbose:
            print("The library was successfully initialized.")

    def set_grid(self, times: numpy.ndarray, thetas: numpy.ndarray, phis: numpy.ndarray, rescale_times: bool = True) -> None:
        """
        Setting the grid.
        - (array) `times` the grid along the time axis
        - (array) `thetas` the grid along the θ axis
        - (array) `phis` the grid along the φ axis
        - (bool) `rescale_times` flag for rescaling the time input
        """

        # IMPORTANT: Only the time component is rescaled as the spatial 
        # grid is between 0 and 2π.

        self.θ_unscaled = thetas
        self.φ_unscaled = phis

        self.dt = numpy.diff(times).mean()
        self.dθ = numpy.diff(thetas).mean()
        self.dφ = numpy.diff(phis).mean()

        if numpy.min(thetas) < 0 or numpy.max(thetas) > π:
            raise Exception("The grid along the θ-axis must be between 0 and π radians.")
        
        if numpy.min(phis) < 0 or numpy.max(phis) > 2 * π:
            raise Exception("The grid along the φ-axis must be between 0 and 2π radians.")
        
        # Rescaling the inputs
        self.tmin, self.tmax = times.min(), times.max()
        self.θmin, self.θmax = thetas.min(), thetas.max()
        self.φmin, self.φmax = phis.min(), phis.max()

        # Rescale into [-1;+1]
        def rescale(x, xmin, xmax):
            return 2.0 * (x - xmin) / (xmax - xmin) - 1.0

        t_rescaled = rescale(times, times.min(), times.max())
        θ_rescaled = rescale(thetas, thetas.min(), thetas.max())
        φ_rescaled = rescale(phis, phis.min(), phis.max())
        
        self.grid = {"times": t_rescaled, "thetas": θ_rescaled, "phis": φ_rescaled}

        if self.verbose:
            print("The grid was successfully set.")

    def set_observation(self, name: str, observation: numpy.ndarray, overwrite: bool = True) -> None:
        """
        Adding an observation.
        - (str) `name` the name of the observation.
        - (array) `obsbervation` the 2D map of the observation.
        - (bool) `overwrite` allows overwriting an observation.
        """
        if name in self.observations and not overwrite:
            raise Exception(f"The observation {name} already exists.")
        
        if not isinstance(observation, (numpy.ndarray)):
            raise Exception("The observation must be an array.")
        
        if len(observation.shape) < 2:
            raise Exception("The observation must have at least 2 dimensions.")
        
        self.observations.update({name: observation.copy()})

        if self.verbose:
            print(f"The observation {name} was successfully added.")

    def set_loss(self, name: str, value: float, overwrite: bool = True) -> None:
        """
        Setting a loss function for the training.
        - (str) `name` the name of the loss function.
        - (float) `value` the weight factor associated.
        - (bool) `overwrite` allows overwriting the weight value
        """
        if name in self.losses and not overwrite:
            raise Exception(f"The loss {name} already exists.")
        
        if not isinstance(value, (int, float)):
            raise Exception("The weight must be a real number.")
        
        self.losses.update({name: value})
        self.history.update({name: []})

        if self.verbose:
            print(f"The loss {name} was successfully added.")

    def set_option(self, name: str, value: bool, overwrite: bool = True):
        """
        Setting a option for the computation of the loss functions.
        - (str) `name` the name of the option (`use_potential_br`, `use_dissipation`).
        - (bool) `value` boolean flag.
        - (bool) `overwrite` allows overwriting the flag value.
        """
        if name in self.options and not overwrite:
            raise Exception(f"The loss {name} already exists.")
        
        if not isinstance(value, (bool)):
            raise Exception("The flag must be a boolean.")
        
        self.options.update({name: value})

        if self.verbose:
            print(f"The option {name} was successfully added.")

    def _array_to_tensor(self, array: numpy.ndarray, requires_grad: bool = True) -> torch.Tensor:
        """
        Converting a numpy array into a torch tensor.
        - (array) `array` the array to convert.
        - (bool) `requires_grad` flag for autograd to record, or not, operations.
        """
        return torch.tensor(array, requires_grad=requires_grad, dtype=torch.float32)
    
    def _autograd(self, tensor: torch.Tensor, inputs: list, retain_graph: bool = True, create_graph: bool = True) -> tuple:
        """
        Computing the gradients of the tensor wrt the inputs.
        - (tensor) `tensor` the tensor from which the gradients are computed.
        - (list) `inputs` the list of inputs.
        - (bool) `retain_graph` keeping in memory the gradient graph.
        - (bool) `create_graph` creating the graph to compute higher-order derivatives.
        - (list) `scales` the scale to be applied to retrieve the unscaled quantity
        """
        return torch.autograd.grad(tensor, inputs, torch.ones_like(tensor), retain_graph, create_graph)
    
    def initialize(self) -> None:
        """
        Initializing the observations before the training.
        """
        for name in self.observations.keys():
            self.tensors.update({
                name: self._array_to_tensor(self.observations[name].reshape(-1, 1))
            })

        times_grid, thetas_grid, phis_grid = numpy.meshgrid(self.grid["times"], self.grid["thetas"], self.grid["phis"], indexing="ij")
        
        # Useful for reshaping torch's squeezed fields later
        self.shape = thetas_grid.shape

        # Initializing the spectral module

        self.spectral = self._spectral(self.θ_unscaled, self.φ_unscaled, 13)

        self.tensors.update({
            "times": self._array_to_tensor(times_grid.reshape(-1, 1)),
            "thetas": self._array_to_tensor(thetas_grid.reshape(-1, 1)),
            "phis": self._array_to_tensor(phis_grid.reshape(-1, 1))
        })

        self.inputs = torch.concat([
            self.tensors["times"], self.tensors["thetas"], self.tensors["phis"]
        ], dim=1)
        
        if self.verbose:
            print("Everything is ready for the training.")

    def warm(self, nb_epochs: int = 10000, lr: float = 1e-3, init: bool = True):
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

        optimizer = torch.optim.AdamW(self.network.parameters(), lr=lr, weight_decay=1e-8)
        scaler = torch.amp.GradScaler("cpu")

        self.tqdm_format = "{percentage:3.2f}% ({remaining} remaining) | Loss = {postfix[0]:.3E}"
        
        with tqdm(total=nb_epochs, bar_format=self.tqdm_format, postfix=[self.lower_loss]) as pg:
            for epoch in range(nb_epochs):
                optimizer.zero_grad(set_to_none=True)

                # Mixed precision to speed up calculations
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
                    loss = self.loss(self.inputs)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)

                scaler.step(optimizer)
                scaler.update()

                with torch.no_grad():
                    if loss.item() < self.lower_loss:
                        torch.save(self.network.state_dict(), "best_model.pt")

                        self.lower_loss = loss.item()
                        pg.postfix[0] = self.lower_loss

                pg.update()

        self.network.load_state_dict(torch.load("best_model.pt"))

    def fine_tune(self, nb_epochs: int = 1000, lr: float = 1e-1):
        """
        Fine-tuning the network with L-BFGS algorithm.
        - (int) `nb_epochs` the number of epochs.
        """
        self.network.train()

        self.network.load_state_dict(torch.load("best_model.pt"))

        optimizer = torch.optim.LBFGS(self.network.parameters(), lr=lr, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad(True)
            loss = self.loss(self.inputs)
            loss.backward()
            return loss
        
        with tqdm(total=nb_epochs, bar_format=self.tqdm_format, postfix=[self.lower_loss]) as pg:
            for epoch in range(nb_epochs):
                loss = optimizer.step(closure)

                with torch.no_grad():
                    if loss.item() < self.lower_loss:
                        torch.save(self.network.state_dict(), "best_model.pt")

                        self.lower_loss = loss.item()
                        pg.postfix[0] = self.lower_loss

                pg.update()
        
        self.network.load_state_dict(torch.load("best_model.pt"))


    def loss(self, inputs: torch.Tensor, evaluate: bool = False) -> torch.Tensor | dict:
        r"""
        Computing the loss function.
        - (tensor) `inputs` the inputs form which the loss is estimated.
        - (bool) `evaluate` returning the fields instead of the loss.
        """
        # Retrieving the predictions
        predictions = self.network(inputs)
        t = 1e0 * predictions[...,0:1]
        s = 1e0 * predictions[...,1:2]
        br = 1e5 * predictions[...,2:3]

        # Retrieving observations
        br_obs = self.tensors["br"]
        dbrdθ_obs = self.tensors["dbrdθ"]
        dbrdφ_obs = self.tensors["dbrdφ"]
        dbrdt_obs = self.tensors["dbrdt"]

        # The scaled inputs
        τ = self.tensors["times"]
        θ = self.tensors["thetas"]
        φ = self.tensors["phis"]

        # Unscaling the inputs
        τ_unscaled = (1 + τ) * (self.tmax - self.tmin) / 2 + self.tmin
        θ_unscaled = (1 + θ) * (self.θmax - self.θmin) / 2 + self.θmin
        φ_unscaled = (1 + φ) * (self.φmax - self.φmin) / 2 + self.φmin

        factor_τ = 2 / (self.tmax - self.tmin)
        factor_θ = 2 / (self.θmax - self.θmin)
        factor_φ = 2 / (self.φmax - self.φmin)

        sinθ = torch.sin(θ_unscaled).clamp(1e-1, 1)
        cosθ = torch.cos(θ_unscaled)

        dtdθ, dtdφ = self._autograd(t, [θ, φ])
        dsdθ, dsdφ = self._autograd(s, [θ, φ])

        # Retrieving uθ and uφ
        uθ = (1 / sinθ) * dtdφ * factor_φ + dsdθ * factor_θ
        uφ = -dtdθ * factor_θ + (1 / sinθ) * dsdφ * factor_φ

        # Computing derivatives
        duθdθ, duθdφ = self._autograd(uθ, [θ,φ])
        duφdθ, duφdφ = self._autograd(uφ, [θ,φ])
        dbrdτ, dbrdθ, dbrdφ = self._autograd(br, [τ,θ,φ])

        # Computing divergence and gradient operators
        divh_uh = (1 / (rC * sinθ)) * (duθdθ * sinθ * factor_θ + uθ * cosθ + duφdφ * factor_φ)
        gradθ_br = (1 / rC) * dbrdθ * factor_θ
        gradφ_br = (1 / (rC * sinθ)) * dbrdφ * factor_φ

        # d2brdθ2 = self._autograd(dbrdθ, [θ])[0]
        # d2brdφ2 = self._autograd(dbrdφ, [φ])[0]

        # Δbr = (1 / (rC**2 * sinθ)) * (cosθ * dbrdθ * factor_θ + sinθ * d2brdθ2 * factor_θ**2) + \
        #     (1 / (rC * sinθ)**2) * d2brdφ2 * factor_φ**2
        
        dbrdt = -(br * divh_uh + gradθ_br * uθ + gradφ_br * uφ)
        
        # Starting the calculation of the losses
        total_loss = torch.tensor([0], dtype=torch.float32)

        dΩ = sinθ.reshape(self.shape) * self.dθ * self.dφ

        lmax_large_scale = 13
        
        if "dbrdt_large_scale" in self.losses:
            xcos, xsin = self.spectral.forward(dbrdt.reshape(self.shape), lmax_large_scale)
            xcos_obs, xsin_obs = self.spectral.forward(dbrdt_obs.reshape(self.shape), lmax_large_scale)
            dbrdt_large_scale = self.spectral.backward(xcos, xsin, lmax_large_scale)
            dbrdt_large_scale_obs = self.spectral.backward(xcos_obs, xsin_obs, lmax_large_scale)
            
            int_Δdbrdt = ((dbrdt_large_scale_obs - dbrdt_large_scale).pow(2) * dΩ).sum(dim=[1,2])
            int_dbrdt = (dbrdt_large_scale_obs.pow(2) * dΩ).sum(dim=[1,2])
            
            loss = self.losses["dbrdt_large_scale"] * (int_Δdbrdt / int_dbrdt).mean()
            self.history["dbrdt_large_scale"].append(loss.detach().numpy())
            total_loss += loss

        if "br_large_scale" in self.losses:
            xcos, xsin = self.spectral.forward(br.reshape(self.shape), lmax_large_scale)
            xcos_obs, xsin_obs = self.spectral.forward(br_obs.reshape(self.shape), lmax_large_scale)
            br_large_scale = self.spectral.backward(xcos, xsin, lmax_large_scale)
            br_large_scale_obs = self.spectral.backward(xcos_obs, xsin_obs, lmax_large_scale)
            
            int_Δbr = ((br_large_scale_obs - br_large_scale).pow(2) * dΩ).sum(dim=[1,2])
            int_br = (br_large_scale_obs.pow(2) * dΩ).sum(dim=[1,2])
            
            loss = self.losses["br_large_scale"] * (int_Δbr / int_br).mean()
            self.history["br_large_scale"].append(loss.detach().numpy())
            total_loss += loss

        if "tangential_geostrophy" in self.losses:
            TG_integral = ((cosθ * divh_uh - sinθ * uθ / rC).reshape(self.shape).pow(2) * dΩ).sum(dim=[1,2]) / torch.sum(dΩ, dim=[1, 2])
            
            loss = self.losses["tangential_geostrophy"] * TG_integral.mean()
            self.history["tangential_geostrophy"].append(loss.detach().numpy())
            total_loss += loss

        if "ΔBr" in self.losses:
            xcos, xsin = self.spectral.forward(dbrdτ.reshape(self.shape), 13)
            dbrdτ_large_scale = self.spectral.backward(xcos, xsin, 13) * factor_τ

            xcos_obs, xsin_obs = self.spectral.forward(dbrdt_obs.reshape(self.shape), 13)
            dbrdt_large_scale_obs = self.spectral.backward(xcos_obs, xsin_obs, 13)

            int_Δdbrdt = ((dbrdt_large_scale_obs - dbrdτ_large_scale).pow(2) * dΩ).sum(dim=[1,2])
            int_dbrdt = (dbrdt_large_scale_obs.pow(2) * dΩ).sum(dim=[1,2])

            loss = self.losses["ΔBr"] * (int_Δdbrdt / int_dbrdt).mean()
            self.history["ΔBr"].append(loss.detach().numpy())
            total_loss += loss

        if "periodicity" in self.losses:
            uθ_r = uθ.reshape(self.shape)
            uφ_r = uφ.reshape(self.shape)
            br_r = br.reshape(self.shape)
            dbrdt_r = dbrdt.reshape(self.shape)

            # Periodicity on fields
            loss_periodicity_uθ = (uθ_r[...,0] - uθ_r[...,-1]).pow(2).mean() / uθ_r[...,0].pow(2).mean()
            loss_periodicity_uφ = (uφ_r[...,0] - uφ_r[...,-1]).pow(2).mean() / uφ_r[...,0].pow(2).mean()
            loss_periodicity_br = (br_r[...,0] - br_r[...,-1]).pow(2).mean() / br_r[...,0].pow(2).mean()
            loss_periodicity_dbrdt = (dbrdt_r[...,0] - dbrdt_r[...,-1]).pow(2).mean() / dbrdt_r[...,0].pow(2).mean()

            divhuh_r = divh_uh.reshape(self.shape)
            gradθbr_r = gradθ_br.reshape(self.shape)
            gradφbr_r = gradφ_br.reshape(self.shape)
            
            # Periodicity on derivatives
            loss_periodicity_divhuh = (divhuh_r[...,0] - divhuh_r[...,-1]).pow(2).mean() / divhuh_r[...,0].pow(2).mean()
            loss_periodicity_gradθbr = (gradθbr_r[...,0] - gradθbr_r[...,-1]).pow(2).mean() / gradθbr_r[...,0].pow(2).mean()
            loss_periodicity_gradφbr = (gradφbr_r[...,0] - gradφbr_r[...,-1]).pow(2).mean() / gradφbr_r[...,0].pow(2).mean()
            
            loss = self.losses["periodicity"] * (loss_periodicity_uθ + loss_periodicity_uφ + loss_periodicity_br + loss_periodicity_dbrdt)
            loss += self.losses["periodicity"] * (loss_periodicity_divhuh + loss_periodicity_gradθbr + loss_periodicity_gradφbr)
            self.history["periodicity"].append(loss.detach().numpy())
            total_loss += loss

        if "strong_norm" in self.losses:

            d2uθdθ2 = self._autograd(duθdθ, [θ])[0]
            d2uθdφ2 = self._autograd(duθdφ, [φ])[0]
            d2uφdθ2 = self._autograd(duφdθ, [θ])[0]
            d2uφdφ2 = self._autograd(duφdφ, [φ])[0]

            Δuθ = (1 / (rC**2 * sinθ)) * (cosθ * duθdθ * factor_θ + sinθ * d2uθdθ2 * factor_θ**2) + \
                (1 / (rC * sinθ)**2) * d2uθdφ2 * factor_φ**2
            
            Δuφ = (1 / (rC**2 * sinθ)) * (cosθ * duφdθ * factor_θ + sinθ * d2uφdθ2 * factor_θ**2) + \
                (1 / (rC * sinθ)**2) * d2uφdφ2 * factor_φ**2
            
            strong_norm = ((Δuθ.reshape(self.shape).pow(2) + Δuφ.reshape(self.shape).pow(2)) * dΩ).sum(dim=[1,2]) / (4 * torch.pi)

            loss = self.losses["strong_norm"] * strong_norm.mean()
            self.history["strong_norm"].append(loss.detach().numpy())
            total_loss += loss

        if "weak_norm" in self.losses:
            weak_norm = ((uθ.reshape(self.shape).pow(2) + uφ.reshape(self.shape).pow(2)) * dΩ).sum(dim=[1,2]) / (4 * torch.pi)

            loss = self.losses["weak_norm"] * weak_norm.mean()
            self.history["weak_norm"].append(loss.detach().numpy())
            total_loss += loss

        if evaluate:
            self.predictions = {
                "br": br.detach().numpy().reshape(self.shape), "t": t.detach().numpy().reshape(self.shape),
                "s": s.detach().numpy().reshape(self.shape), "uθ": uθ.detach().numpy().reshape(self.shape),
                "uφ": uφ.detach().numpy().reshape(self.shape), "dbrdt": dbrdt.detach().numpy().reshape(self.shape)
            }
            return self.predictions

        return total_loss
    
    def evaluate(self) -> dict | torch.Tensor:
        """
        Evaluating the networks predictions.
        """
        self.network.load_state_dict(torch.load("best_model.pt"))
        self.network.eval()

        return self.loss(self.inputs, True)