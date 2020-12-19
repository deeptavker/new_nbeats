# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
N-BEATS Model.
"""
from typing import Tuple

import numpy as np
import torch as t


class CNNBlock(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = t.nn.ModuleList([t.nn.Conv1d(1, 5, 5, 1, 2)])

    def forward(self, x):
        conv_inp = x.unsqueeze(1)
        out = self.layers[0](conv_inp)
        out = out.transpose(0,1)
        return out

class MLP(t.nn.Module):
    def __init__(self, input_size :int, output_size: int):
        super().__init__()
        self.layers = t.nn.ModuleList([t.nn.Linear(in_features= input_size, out_features= output_size)])
        
    def forward(self, x):
        activation = self.layers[0](x)
        activation = t.relu(activation)
        return activation
    

class NBeatsBlock(t.nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """
    def __init__(self,
                 input_size,
                 theta_size: int,
                 basis_function: t.nn.Module,
                 layers: int,
                 layer_size: int):
        """
        N-BEATS block.

        :param input_size: Insample size.
        :param theta_size:  Number of parameters for the basis function.
        :param basis_function: Basis function which takes the parameters and produces backcast and forecast.
        :param layers: Number of layers.
        :param layer_size: Layer size.
        """
        super().__init__()
        self.layers = t.nn.ModuleList([t.nn.Linear(in_features=input_size, out_features=layer_size)] +
                                      [t.nn.Linear(in_features=layer_size, out_features=layer_size)
                                       for _ in range(layers - 1)])
        self.basis_parameters = t.nn.Linear(in_features=layer_size, out_features=theta_size)
        self.basis_function = basis_function

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        block_input = x
        for layer in self.layers:
            block_input = t.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)


class NBeats(t.nn.Module):
    """
    N-Beats Model.
    """
    def __init__(self, blocks: t.nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, x: t.Tensor, input_mask: t.Tensor) -> t.Tensor:
        
        input_mask = input_mask.flip(dims=(1,))
        
        x1, x2, x3, x4, x5 = self.blocks[-2](x)
        r1, r2, r3, r4, r5 = x1.flip(dims=(1,)), x2.flip(dims=(1,)), x3.flip(dims=(1,)), x4.flip(dims=(1,)), x5.flip(dims=(1,))
        f1, f2, f3, f4, f5 = x1[:, -1:], x2[:, -1:], x3[:, -1:], x4[:, -1:], x5[:, -1:]


        nb_blocks = self.blocks[:-2]
        nb = len(nb_blocks)

        blocks1 = nb_blocks[:nb//5]
        blocks2 = nb_blocks[nb//5:nb*2//5]
        blocks3 = nb_blocks[nb*2//5:nb*3//5]
        blocks4 = nb_blocks[nb*3//5: nb*4//5 ]
        blocks5 = nb_blocks[nb*4//5:]

        for i, block in enumerate(blocks1):
            backcast, block_forecast = block(r1)
            r1 = (r1 - backcast) * input_mask
            f1 = f1 + block_forecast

        for i, block in enumerate(blocks2):
            backcast, block_forecast = block(r2)
            r2 = (r2 - backcast) * input_mask
            f2 = f2 + block_forecast

        for i, block in enumerate(blocks3):
            backcast, block_forecast = block(r3)
            r3 = (r3 - backcast) * input_mask
            f3 = f3 + block_forecast

        for i, block in enumerate(blocks4):
            backcast, block_forecast = block(r4)
            r4 = (r4 - backcast) * input_mask
            f4 = f4 + block_forecast
            
        for i, block in enumerate(blocks5):
            backcast, block_forecast = block(r5)
            r5 = (r5 - backcast) * input_mask
            f5 = f5 + block_forecast
        
        f= t.cat((f1,f2,f3, f4, f5),dim=1)
        forecast = self.blocks[-1](f)

        return forecast


class GenericBasis(t.nn.Module):
    """
    Generic basis function.
    """
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: t.Tensor):
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]


class TrendBasis(t.nn.Module):
    """
    Polynomial function to model trend.
    """
    def __init__(self, degree_of_polynomial: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.polynomial_size = degree_of_polynomial + 1  # degree of polynomial with constant term
        self.backcast_time = t.nn.Parameter(
            t.tensor(np.concatenate([np.power(np.arange(backcast_size, dtype=np.float) / backcast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=t.float32),
            requires_grad=False)
        self.forecast_time = t.nn.Parameter(
            t.tensor(np.concatenate([np.power(np.arange(forecast_size, dtype=np.float) / forecast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=t.float32), requires_grad=False)

    def forward(self, theta: t.Tensor):
        backcast = t.einsum('bp,pt->bt', theta[:, self.polynomial_size:], self.backcast_time)
        forecast = t.einsum('bp,pt->bt', theta[:, :self.polynomial_size], self.forecast_time)
        return backcast, forecast


class SeasonalityBasis(t.nn.Module):
    """
    Harmonic functions to model seasonality.
    """
    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.frequency = np.append(np.zeros(1, dtype=np.float32),
                                   np.arange(harmonics, harmonics / 2 * forecast_size,
                                             dtype=np.float32) / harmonics)[None, :]
        backcast_grid = -2 * np.pi * (
                np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency
        forecast_grid = 2 * np.pi * (
                np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency
        self.backcast_cos_template = t.nn.Parameter(t.tensor(np.transpose(np.cos(backcast_grid)), dtype=t.float32),
                                                    requires_grad=False)
        self.backcast_sin_template = t.nn.Parameter(t.tensor(np.transpose(np.sin(backcast_grid)), dtype=t.float32),
                                                    requires_grad=False)
        self.forecast_cos_template = t.nn.Parameter(t.tensor(np.transpose(np.cos(forecast_grid)), dtype=t.float32),
                                                    requires_grad=False)
        self.forecast_sin_template = t.nn.Parameter(t.tensor(np.transpose(np.sin(forecast_grid)), dtype=t.float32),
                                                    requires_grad=False)

    def forward(self, theta: t.Tensor):
        params_per_harmonic = theta.shape[1] // 4
        backcast_harmonics_cos = t.einsum('bp,pt->bt', theta[:, 2 * params_per_harmonic:3 * params_per_harmonic],
                                          self.backcast_cos_template)
        backcast_harmonics_sin = t.einsum('bp,pt->bt', theta[:, 3 * params_per_harmonic:], self.backcast_sin_template)
        backcast = backcast_harmonics_sin + backcast_harmonics_cos
        forecast_harmonics_cos = t.einsum('bp,pt->bt',
                                          theta[:, :params_per_harmonic], self.forecast_cos_template)
        forecast_harmonics_sin = t.einsum('bp,pt->bt', theta[:, params_per_harmonic:2 * params_per_harmonic],
                                          self.forecast_sin_template)
        forecast = forecast_harmonics_sin + forecast_harmonics_cos

        return backcast, forecast
