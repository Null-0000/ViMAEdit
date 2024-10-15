from diffusers import DDIMScheduler, DPMSolverMultistepScheduler
import torch
from typing import Optional

from diffusers.utils.torch_utils import randn_tensor


def compute_mu(k_t, alpha_prod_t, alpha_prod_prev_t):
    a = (alpha_prod_prev_t * (1 - alpha_prod_t)).sqrt() / (alpha_prod_t * (1 - alpha_prod_prev_t)).sqrt()
    b = 1 - (k_t * (1 - a) + a)**2
    c = b * (1 - alpha_prod_t) / (1 - alpha_prod_t/alpha_prod_prev_t)
    return c.sqrt()


class EDDIMScheduler(DDIMScheduler):
    def __init__(self, eta, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eta = eta

    def edit_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        source_original_sample: torch.FloatTensor,
        sample: torch.FloatTensor,
        editing_mask: torch.FloatTensor = None,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
    ):
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        assert self.config.prediction_type == "epsilon"
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        if source_original_sample is not None:
            pred_original_sample = source_original_sample + pred_original_sample - pred_original_sample[:1]

        # 4. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        if editing_mask is None:
            std_dev_t = 0
        else:
            std_dev_t = self.eta * editing_mask.to(dtype=model_output.dtype) * variance ** (0.5)

        b_t = ((1 - alpha_prod_t_prev - std_dev_t ** 2) / (1 - alpha_prod_t)).sqrt()
        prev_sample = (alpha_prod_t_prev.sqrt() - alpha_prod_t.sqrt() * b_t) * pred_original_sample + b_t * sample

        if (editing_mask is not None) and (editing_mask.sum() > 0):
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                variance_noise = randn_tensor(
                    (1, *model_output.shape[1:]), generator=generator, device=model_output.device, dtype=model_output.dtype
                ).repeat(model_output.shape[0], 1, 1, 1)
            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance

        return prev_sample


class EDPMSolverMultistepScheduler(DPMSolverMultistepScheduler):
    def convert_model_output(
        self,
        model_output: torch.FloatTensor,
        *args,
        sample: torch.FloatTensor = None,
        **kwargs,
    ) -> torch.FloatTensor:
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError("missing `sample` as a required keyward argument")

        # DPM-Solver++ needs to solve an integral of the data prediction model.
        assert self.config.algorithm_type in ["dpmsolver++", "sde-dpmsolver++"] and self.config.prediction_type == "epsilon"
        sigma = self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        x0_pred = (sample - sigma_t * model_output) / alpha_t
        if ('src_x0' in kwargs) and (kwargs['src_x0'] is not None):
            x0_pred = kwargs['src_x0'] + x0_pred - x0_pred[:1]
        if self.config.thresholding:
            x0_pred = self._threshold_sample(x0_pred)
        return x0_pred

    # copy from scheduling_dpmsolver_multistep.py
    def edit_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        source_original_sample: torch.FloatTensor,
        sample: torch.FloatTensor,
        editing_mask: torch.FloatTensor = None,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
    ):
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Improve numerical stability for small number of steps
        lower_order_final = (self.step_index == len(self.timesteps) - 1) and (
            self.config.euler_at_final
            or (self.config.lower_order_final and len(self.timesteps) < 15)
            or self.config.final_sigmas_type == "zero"
        )
        lower_order_second = (
            (self.step_index == len(self.timesteps) - 2) and self.config.lower_order_final and len(self.timesteps) < 15
        )

        model_output = self.convert_model_output(model_output, sample=sample, src_x0=source_original_sample)
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)
        if variance_noise is None:
            noise = randn_tensor(
                (1, *model_output.shape[1:]), generator=generator, device=model_output.device, dtype=torch.float32
            ).repeat(model_output.shape[0], 1, 1, 1)
        else:
            noise = variance_noise.to(device=model_output.device, dtype=torch.float32).repeat(model_output.shape[0], 1, 1, 1)

        if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            self.config.algorithm_type = "dpmsolver++"
            prev_sample = self.dpm_solver_first_order_update(model_output, sample=sample, noise=noise)
            if editing_mask is not None:
                self.config.algorithm_type = "sde-dpmsolver++"
                prev_sample_sde = self.dpm_solver_first_order_update(model_output, sample=sample, noise=noise)
                editing_mask = editing_mask.to(dtype=sample.dtype)
                prev_sample = editing_mask * prev_sample_sde + (1 - editing_mask) * prev_sample

        elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
            self.config.algorithm_type = "dpmsolver++"
            prev_sample = self.multistep_dpm_solver_second_order_update(self.model_outputs, sample=sample, noise=noise)
            if editing_mask is not None:
                self.config.algorithm_type = "sde-dpmsolver++"
                prev_sample_sde = self.multistep_dpm_solver_second_order_update(self.model_outputs, sample=sample, noise=noise)
                editing_mask = editing_mask.to(dtype=sample.dtype)
                prev_sample = editing_mask * prev_sample_sde + (1 - editing_mask) * prev_sample
        else:
            self.config.algorithm_type = "dpmsolver++"
            prev_sample = self.multistep_dpm_solver_third_order_update(self.model_outputs, sample=sample)
            if editing_mask is not None:
                self.config.algorithm_type = "sde-dpmsolver++"
                prev_sample_sde = self.multistep_dpm_solver_third_order_update(self.model_outputs, sample=sample)
                editing_mask = editing_mask.to(dtype=sample.dtype)
                prev_sample = editing_mask * prev_sample_sde + (1 - editing_mask) * prev_sample

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        # Cast sample back to expected dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        return prev_sample