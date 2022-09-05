# Copyright 2022 NVIDIA and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Union

import numpy as np
import paddle

from ..configuration_utils import ConfigMixin
from ..configuration_utils import register_to_config
from .scheduling_utils import SchedulerMixin


class KarrasVeScheduler(SchedulerMixin, ConfigMixin):
    """
    Stochastic sampling from Karras et al. [1] tailored to the Variance-Expanding (VE) models [2]. Use Algorithm 2 and
    the VE column of Table 1 from [1] for reference.

    [1] Karras, Tero, et al. "Elucidating the Design Space of Diffusion-Based Generative Models."
    https://arxiv.org/abs/2206.00364 [2] Song, Yang, et al. "Score-based generative modeling through stochastic
    differential equations." https://arxiv.org/abs/2011.13456
    """

    @register_to_config
    def __init__(
        self,
        sigma_min=0.02,
        sigma_max=100,
        s_noise=1.007,
        s_churn=80,
        s_min=0.05,
        s_max=50,
        tensor_format="pd",
    ):
        """
        For more details on the parameters, see the original paper's Appendix E.: "Elucidating the Design Space of
        Diffusion-Based Generative Models." https://arxiv.org/abs/2206.00364. The grid search values used to find the
        optimal {s_noise, s_churn, s_min, s_max} for a specific model are described in Table 5 of the paper.

        Args:
            sigma_min (`float`): minimum noise magnitude
            sigma_max (`float`): maximum noise magnitude
            s_noise (`float`): the amount of additional noise to counteract loss of detail during sampling.
                A reasonable range is [1.000, 1.011].
            s_churn (`float`): the parameter controlling the overall amount of stochasticity.
                A reasonable range is [0, 100].
            s_min (`float`): the start value of the sigma range where we add noise (enable stochasticity).
                A reasonable range is [0, 10].
            s_max (`float`): the end value of the sigma range where we add noise.
                A reasonable range is [0.2, 80].
        """
        # setable values
        self.num_inference_steps = None
        self.timesteps = None
        self.schedule = None  # sigma(t_i)

        self.tensor_format = tensor_format
        self.set_format(tensor_format=tensor_format)

    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        self.timesteps = np.arange(0, self.num_inference_steps)[::-1].copy()
        self.schedule = [(self.sigma_max * (self.sigma_min**2 / self.sigma_max**2)**(i / (num_inference_steps - 1)))
                         for i in self.timesteps]
        self.schedule = np.array(self.schedule, dtype=np.float32)

        self.set_format(tensor_format=self.tensor_format)

    def add_noise_to_input(self, sample, sigma, generator=None):
        """
        Explicit Langevin-like "churn" step of adding noise to the sample according to a factor gamma_i ≥ 0 to reach a
        higher noise level sigma_hat = sigma_i + gamma_i*sigma_i.
        """
        if self.s_min <= sigma <= self.s_max:
            gamma = min(self.s_churn / self.num_inference_steps, 2**0.5 - 1)
        else:
            gamma = 0

        # sample eps ~ N(0, S_noise^2 * I)
        eps = self.s_noise * paddle.randn(sample.shape)
        sigma_hat = sigma + gamma * sigma
        sample_hat = sample + ((sigma_hat**2 - sigma**2)**0.5 * eps)

        return sample_hat, sigma_hat

    def step(
        self,
        model_output: Union[paddle.Tensor, np.ndarray],
        sigma_hat: float,
        sigma_prev: float,
        sample_hat: Union[paddle.Tensor, np.ndarray],
    ):
        pred_original_sample = sample_hat + sigma_hat * model_output
        derivative = (sample_hat - pred_original_sample) / sigma_hat
        sample_prev = sample_hat + (sigma_prev - sigma_hat) * derivative

        return {"prev_sample": sample_prev, "derivative": derivative}

    def step_correct(
        self,
        model_output: Union[paddle.Tensor, np.ndarray],
        sigma_hat: float,
        sigma_prev: float,
        sample_hat: Union[paddle.Tensor, np.ndarray],
        sample_prev: Union[paddle.Tensor, np.ndarray],
        derivative: Union[paddle.Tensor, np.ndarray],
    ):
        pred_original_sample = sample_prev + sigma_prev * model_output
        derivative_corr = (sample_prev - pred_original_sample) / sigma_prev
        sample_prev = sample_hat + (sigma_prev - sigma_hat) * (0.5 * derivative + 0.5 * derivative_corr)
        return {"prev_sample": sample_prev, "derivative": derivative_corr}

    def add_noise(self, original_samples, noise, timesteps):
        raise NotImplementedError()
