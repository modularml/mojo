# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""UniPC multistep scheduler for diffusion models (numpy-only).

Implements the UniPC-BH2 algorithm with corrector and predictor steps,
compatible with flow-matching diffusion models. This is a numpy-only port
of the diffusers UniPCMultistepScheduler, specialized for the Wan pipeline
configuration.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


class UniPCMultistepScheduler:
    """NumPy-only UniPC multistep scheduler for diffusion models.

    Implements the UniPC (Unified Predictor-Corrector) framework with B(h)
    updates for fast sampling of diffusion models. Supports flow-matching
    prediction type and the BH2 solver variant.

    This scheduler is designed for use with the Wan 2.2 T2V pipeline.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        solver_order: int = 2,
        prediction_type: str = "flow_prediction",
        predict_x0: bool = True,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
        disable_corrector: list[int] | None = None,
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        use_flow_sigmas: bool = False,
        flow_shift: float = 1.0,
        time_shift_type: str = "exponential",
        final_sigmas_type: str = "zero",
        # Keep backward-compatible kwargs from the old interface
        order: int = 1,
        **unused_kwargs,
    ) -> None:
        self.num_train_timesteps = int(num_train_timesteps)
        self.solver_order = int(solver_order)
        self.prediction_type = prediction_type
        self.predict_x0 = predict_x0
        self.solver_type = solver_type
        self.lower_order_final = lower_order_final
        self.disable_corrector = disable_corrector or []
        self.thresholding = thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.sample_max_value = sample_max_value
        self.use_flow_sigmas = use_flow_sigmas
        self.flow_shift = float(flow_shift)
        self.time_shift_type = time_shift_type
        self.final_sigmas_type = final_sigmas_type
        self.order = int(order)

        if solver_type not in ("bh1", "bh2"):
            raise NotImplementedError(
                f"solver_type={solver_type} is not implemented"
            )

        # Internal state — initialized by set_timesteps()
        self.num_inference_steps: int | None = None
        self.timesteps: npt.NDArray[np.float64] | None = None
        self.sigmas: npt.NDArray[np.float64] | None = None
        self.model_outputs: list[npt.NDArray[np.float64] | None] = [
            None
        ] * solver_order
        self.timestep_list: list[float | None] = [None] * solver_order
        self.lower_order_nums: int = 0
        self.last_sample: npt.NDArray[np.float64] | None = None
        self._step_index: int | None = None
        self._begin_index: int | None = None
        self.this_order: int = 1

    @property
    def step_index(self) -> int | None:
        """The index counter for the current timestep."""
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        """The index for the first timestep."""
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        """Sets the begin index for the scheduler."""
        self._begin_index = begin_index

    def set_timesteps(
        self, num_inference_steps: int, flow_shift: float | None = None
    ) -> None:
        """Initialize internal state for a denoising run.

        Must be called before the first ``step()`` call.

        Args:
            num_inference_steps: Number of denoising steps.
            flow_shift: Per-call flow-matching shift. When ``None``, the
                shift from the scheduler config is used.
        """
        shift = self.flow_shift if flow_shift is None else float(flow_shift)
        if self.use_flow_sigmas:
            # Match diffusers UniPCMultistepScheduler.set_timesteps exactly:
            # linspace over sigmas directly (not 1-alphas), drop last before shifting.
            sigmas = np.linspace(
                1, 1 / self.num_train_timesteps, num_inference_steps + 1
            )[:-1]
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
            # Clamp sigma[0] just below 1.0 so timestep < num_train_timesteps
            # (matches diffusers' `sigmas[0] -= eps` guard).
            eps = 1.0 / self.num_train_timesteps
            if sigmas[0] >= 1.0:
                sigmas[0] -= eps
            timesteps = (sigmas * self.num_train_timesteps).copy()

            if self.final_sigmas_type == "sigma_min":
                sigma_last = float(sigmas[-1])
            elif self.final_sigmas_type == "zero":
                sigma_last = 0.0
            else:
                raise ValueError(
                    f"final_sigmas_type must be 'zero' or 'sigma_min', "
                    f"got {self.final_sigmas_type}"
                )
            sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)
        else:
            timesteps = np.linspace(
                0,
                self.num_train_timesteps - 1,
                num_inference_steps + 1,
                dtype=np.float64,
            )
            timesteps = np.round(timesteps)[::-1][:-1].copy().astype(np.int64)
            # Compute sigmas from beta schedule (VP-type)
            betas = np.linspace(0.0001, 0.02, self.num_train_timesteps)
            alphas = 1.0 - betas
            alphas_cumprod = np.cumprod(alphas)
            all_sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
            sigmas = np.interp(
                timesteps, np.arange(len(all_sigmas)), all_sigmas
            )
            if self.final_sigmas_type == "zero":
                sigma_last = 0.0
            else:
                sigma_last = float(sigmas[-1])
            sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.sigmas = sigmas.astype(np.float64)
        self.timesteps = timesteps.astype(np.int64)
        self.num_inference_steps = len(timesteps)

        # Reset solver state
        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self._step_index = None
        self._begin_index = None
        self.this_order = 1

    def retrieve_timesteps_and_sigmas(
        self,
        image_seq_len: int,
        num_inference_steps: int,
        reverse: bool = False,
        sigma_min: float | None = None,
        flow_shift: float | None = None,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Build timestep/sigma schedule, with flow-matching support.

        Also calls ``set_timesteps`` so the scheduler is ready for stepping.

        Args:
            image_seq_len: Sequence length (unused for flow-matching).
            num_inference_steps: Number of denoising steps.
            reverse: Whether to reverse timesteps (non-flow only).
            sigma_min: Unused. Accepted for interface compatibility with
                other schedulers.
            flow_shift: Per-call flow-matching shift. When ``None``, the
                shift from the scheduler config is used.

        Returns:
            Tuple of (timesteps, sigmas) as float32 arrays.
        """
        del image_seq_len, sigma_min
        self.set_timesteps(num_inference_steps, flow_shift=flow_shift)
        assert self.timesteps is not None
        assert self.sigmas is not None

        if self.use_flow_sigmas:
            # Use the int64-rounded timesteps from set_timesteps (matching
            # diffusers), cast to float32 for the pipeline.
            timesteps = self.timesteps.astype(np.float32)
            sigmas = self.sigmas.astype(np.float32)
        else:
            if reverse:
                timesteps = (
                    (
                        float(self.num_train_timesteps)
                        - self.timesteps.astype(np.float64)
                    )
                    / float(self.num_train_timesteps)
                ).astype(np.float32)
            else:
                timesteps = (
                    self.timesteps.astype(np.float64)
                    / float(self.num_train_timesteps)
                ).astype(np.float32)
            sigmas = self.sigmas.astype(np.float32)

        return timesteps, sigmas

    def _sigma_to_alpha_sigma_t(
        self, sigma: float | np.float64
    ) -> tuple[np.float64, np.float64]:
        """Convert sigma to (alpha_t, sigma_t).

        For flow matching: alpha_t = 1 - sigma, sigma_t = sigma.
        For VP-type: alpha_t = 1/sqrt(sigma^2+1), sigma_t = sigma*alpha_t.
        """
        if self.use_flow_sigmas:
            alpha_t = np.float64(1.0) - np.float64(sigma)
            sigma_t = np.float64(sigma)
        else:
            sigma_val = np.float64(sigma)
            alpha_t = np.float64(1.0) / np.sqrt(sigma_val**2 + 1.0)
            sigma_t = sigma_val * alpha_t
        return alpha_t, sigma_t

    def _init_step_index(self, timestep: float) -> None:
        """Initialize the step_index counter for the scheduler."""
        if self._begin_index is None:
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def index_for_timestep(self, timestep: float) -> int:
        """Find the index for a given timestep in the schedule."""
        assert self.timesteps is not None
        indices = np.where(
            np.isclose(self.timesteps, float(timestep), atol=0.5)
        )[0]
        if len(indices) == 0:
            return len(self.timesteps) - 1
        elif len(indices) > 1:
            return int(indices[1])
        else:
            return int(indices[0])

    def convert_model_output(
        self,
        model_output: np.ndarray,
        sample: np.ndarray,
    ) -> np.ndarray:
        """Convert the raw model output to x0 prediction.

        For flow_prediction: x0 = sample - sigma_t * model_output.

        Args:
            model_output: Direct output from the diffusion model.
            sample: Current noisy sample.

        Returns:
            Converted output (x0 prediction when predict_x0=True).
        """
        assert self._step_index is not None
        assert self.sigmas is not None
        sigma = self.sigmas[self._step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)

        if self.predict_x0:
            if self.prediction_type == "epsilon":
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.prediction_type == "sample":
                x0_pred = model_output
            elif self.prediction_type == "v_prediction":
                x0_pred = alpha_t * sample - sigma_t * model_output
            elif self.prediction_type == "flow_prediction":
                x0_pred = sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type={self.prediction_type} not supported"
                )
            return x0_pred
        else:
            if self.prediction_type == "epsilon":
                return model_output
            elif self.prediction_type == "sample":
                return (sample - alpha_t * model_output) / sigma_t
            elif self.prediction_type == "v_prediction":
                return alpha_t * model_output + sigma_t * sample
            else:
                raise ValueError(
                    f"prediction_type={self.prediction_type} not supported"
                )

    def multistep_uni_p_bh_update(
        self,
        model_output: np.ndarray,
        sample: np.ndarray,
        order: int,
    ) -> np.ndarray:
        """One predictor step for UniP (B(h) version).

        Args:
            model_output: Direct (non-converted) model output.
            sample: Current sample.
            order: Solver order for this step.

        Returns:
            Predicted sample at the next timestep.
        """
        assert self._step_index is not None
        assert self.sigmas is not None
        model_output_list = self.model_outputs

        m0 = model_output_list[-1]
        assert m0 is not None
        x = sample

        sigma_t_raw = float(self.sigmas[self._step_index + 1])
        sigma_s0_raw = float(self.sigmas[self._step_index])
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t_raw)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0_raw)

        with np.errstate(divide="ignore"):
            lambda_t = np.log(alpha_t) - np.log(sigma_t)
            lambda_s0 = np.log(alpha_s0) - np.log(sigma_s0)
        h = lambda_t - lambda_s0

        rks_list: list[float | np.float64] = []
        D1s = []
        for i in range(1, order):
            si = self._step_index - i
            mi = model_output_list[-(i + 1)]
            assert mi is not None
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(
                float(self.sigmas[si])
            )
            lambda_si = np.log(alpha_si) - np.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks_list.append(rk)
            D1s.append((mi - m0) / rk)

        rks_list.append(1.0)
        rks_arr = np.array(rks_list, dtype=np.float64)

        hh = -h if self.predict_x0 else h
        h_phi_1 = np.expm1(hh)  # e^hh - 1
        h_phi_k = h_phi_1 / hh - 1.0

        factorial_i = 1

        if self.solver_type == "bh1":
            B_h = hh
        elif self.solver_type == "bh2":
            B_h = np.expm1(hh)
        else:
            raise NotImplementedError(f"solver_type={self.solver_type}")

        R_list = []
        b_list: list[np.float64] = []
        for i in range(1, order + 1):
            R_list.append(np.power(rks_arr, i - 1))
            b_list.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1.0 / factorial_i

        R_arr = np.stack(R_list)  # (order, len(rks))
        b_arr = np.array(b_list, dtype=np.float64)

        if len(D1s) > 0:
            D1s_arr = np.stack(D1s, axis=1)  # (B, K, ...)
            if order == 2:
                rhos_p = np.array([0.5], dtype=np.float64)
            else:
                rhos_p = np.linalg.solve(R_arr[:-1, :-1], b_arr[:-1])
        else:
            D1s_arr = None

        if self.predict_x0:
            x_t_ = (sigma_t / sigma_s0) * x - alpha_t * h_phi_1 * m0
            if D1s_arr is not None:
                pred_res = np.einsum("k,bk...->b...", rhos_p, D1s_arr)
            else:
                pred_res = 0.0
            x_t = x_t_ - alpha_t * B_h * pred_res
        else:
            x_t_ = (alpha_t / alpha_s0) * x - sigma_t * h_phi_1 * m0
            if D1s_arr is not None:
                pred_res = np.einsum("k,bk...->b...", rhos_p, D1s_arr)
            else:
                pred_res = 0.0
            x_t = x_t_ - sigma_t * B_h * pred_res

        return x_t

    def multistep_uni_c_bh_update(
        self,
        this_model_output: np.ndarray,
        last_sample: np.ndarray,
        this_sample: np.ndarray,
        order: int,
    ) -> np.ndarray:
        """One corrector step for UniC (B(h) version).

        Args:
            this_model_output: Converted model output at current step.
            last_sample: Sample before the last predictor step.
            this_sample: Sample after the last predictor step.
            order: Corrector order.

        Returns:
            Corrected sample.
        """
        assert self._step_index is not None
        assert self.sigmas is not None
        model_output_list = self.model_outputs

        m0 = model_output_list[-1]
        assert m0 is not None
        x = last_sample
        model_t = this_model_output

        sigma_t_raw = float(self.sigmas[self._step_index])
        sigma_s0_raw = float(self.sigmas[self._step_index - 1])
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t_raw)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0_raw)

        with np.errstate(divide="ignore"):
            lambda_t = np.log(alpha_t) - np.log(sigma_t)
            lambda_s0 = np.log(alpha_s0) - np.log(sigma_s0)
        h = lambda_t - lambda_s0

        rks_list: list[float | np.float64] = []
        D1s = []
        for i in range(1, order):
            si = self._step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            assert mi is not None
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(
                float(self.sigmas[si])
            )
            lambda_si = np.log(alpha_si) - np.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks_list.append(rk)
            D1s.append((mi - m0) / rk)

        rks_list.append(1.0)
        rks_arr = np.array(rks_list, dtype=np.float64)

        hh = -h if self.predict_x0 else h
        h_phi_1 = np.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1.0

        factorial_i = 1

        if self.solver_type == "bh1":
            B_h = hh
        elif self.solver_type == "bh2":
            B_h = np.expm1(hh)
        else:
            raise NotImplementedError(f"solver_type={self.solver_type}")

        R_list = []
        b_list: list[np.float64] = []
        for i in range(1, order + 1):
            R_list.append(np.power(rks_arr, i - 1))
            b_list.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1.0 / factorial_i

        R_arr = np.stack(R_list)
        b_arr = np.array(b_list, dtype=np.float64)

        if len(D1s) > 0:
            D1s_arr = np.stack(D1s, axis=1)
        else:
            D1s_arr = None

        # For order 1, use simplified rhos_c = [0.5]
        if order == 1:
            rhos_c = np.array([0.5], dtype=np.float64)
        else:
            rhos_c = np.linalg.solve(R_arr, b_arr)

        if self.predict_x0:
            x_t_ = (sigma_t / sigma_s0) * x - alpha_t * h_phi_1 * m0
            if D1s_arr is not None:
                corr_res = np.einsum("k,bk...->b...", rhos_c[:-1], D1s_arr)
            else:
                corr_res = 0.0
            D1_t = model_t - m0
            x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        else:
            x_t_ = (alpha_t / alpha_s0) * x - sigma_t * h_phi_1 * m0
            if D1s_arr is not None:
                corr_res = np.einsum("k,bk...->b...", rhos_c[:-1], D1s_arr)
            else:
                corr_res = 0.0
            D1_t = model_t - m0
            x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)

        return x_t

    def step(
        self,
        model_output: np.ndarray,
        timestep: float,
        sample: np.ndarray,
    ) -> np.ndarray:
        """Predict the sample at the previous timestep using UniPC.

        Orchestrates the corrector (UniC) and predictor (UniP) updates.

        Args:
            model_output: Direct output from the learned diffusion model.
            timestep: Current discrete timestep.
            sample: Current noisy sample.

        Returns:
            Denoised sample at the next timestep.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "num_inference_steps is None — call set_timesteps() first"
            )
        assert self.timesteps is not None
        assert self.sigmas is not None

        if self._step_index is None:
            self._init_step_index(float(timestep))
        assert self._step_index is not None

        # Ensure float64 for numerical precision
        model_output = np.asarray(model_output, dtype=np.float64)
        sample = np.asarray(sample, dtype=np.float64)

        use_corrector = (
            self._step_index > 0
            and (self._step_index - 1) not in self.disable_corrector
            and self.last_sample is not None
        )

        model_output_convert = self.convert_model_output(
            model_output, sample=sample
        )

        if use_corrector:
            assert self.last_sample is not None
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )

        # Shift model output history
        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]

        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = float(timestep)

        # Determine order for this predictor step
        if self.lower_order_final:
            this_order = min(
                self.solver_order,
                len(self.timesteps) - self._step_index,
            )
        else:
            this_order = self.solver_order

        # Warmup: can't use higher order than available history
        self.this_order = min(this_order, self.lower_order_nums + 1)
        assert self.this_order > 0

        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(
            model_output=model_output,
            sample=sample,
            order=self.this_order,
        )

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1

        self._step_index += 1
        return prev_sample

    def _lambda_from_sigma(self, sigma: float) -> float:
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        with np.errstate(divide="ignore"):
            return float(np.log(alpha_t) - np.log(sigma_t))

    def _predictor_coefficients(
        self, step_index: int, order: int
    ) -> tuple[float, float, float]:
        assert self.sigmas is not None
        sigma_t_raw = float(self.sigmas[step_index + 1])
        sigma_s0_raw = float(self.sigmas[step_index])
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t_raw)
        _, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0_raw)

        lambda_t = self._lambda_from_sigma(sigma_t_raw)
        lambda_s0 = self._lambda_from_sigma(sigma_s0_raw)
        h = lambda_t - lambda_s0
        b_h = float(np.expm1(-h))
        sample_scale = float(sigma_t / sigma_s0)
        m0_scale = float(-alpha_t * b_h)
        m1_scale = 0.0

        if order == 2:
            sigma_si_raw = float(self.sigmas[step_index - 1])
            lambda_si = self._lambda_from_sigma(sigma_si_raw)
            rk = (lambda_si - lambda_s0) / h
            m1_scale = float(-alpha_t * b_h * 0.5 / rk)
            m0_scale -= m1_scale

        return sample_scale, m0_scale, m1_scale

    def _corrector_coefficients(
        self, step_index: int, order: int
    ) -> tuple[float, float, float, float]:
        assert self.sigmas is not None
        sigma_t_raw = float(self.sigmas[step_index])
        sigma_s0_raw = float(self.sigmas[step_index - 1])
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t_raw)
        _, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0_raw)

        lambda_t = self._lambda_from_sigma(sigma_t_raw)
        lambda_s0 = self._lambda_from_sigma(sigma_s0_raw)
        h = lambda_t - lambda_s0
        hh = -h
        h_phi_1 = float(np.expm1(hh))
        b_h = float(np.expm1(hh))
        sample_scale = float(sigma_t / sigma_s0)

        if order == 1:
            shared = float(-alpha_t * b_h * 0.5)
            return sample_scale, shared, 0.0, shared

        sigma_si_raw = float(self.sigmas[step_index - 2])
        lambda_si = self._lambda_from_sigma(sigma_si_raw)
        rk = (lambda_si - lambda_s0) / h

        rks = np.array([rk, 1.0], dtype=np.float64)
        h_phi_k = h_phi_1 / hh - 1.0
        factorial_i = 1
        r_matrix = []
        b_vector = []
        for i in range(1, order + 1):
            r_matrix.append(np.power(rks, i - 1))
            b_vector.append(h_phi_k * factorial_i / b_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1.0 / factorial_i
        rhos_c = np.linalg.solve(
            np.stack(r_matrix), np.array(b_vector, dtype=np.float64)
        )

        m1_scale = float(-alpha_t * b_h * rhos_c[0] / rk)
        m0_scale = float(
            -alpha_t * h_phi_1 + alpha_t * b_h * (rhos_c[0] / rk + rhos_c[-1])
        )
        mt_scale = float(-alpha_t * b_h * rhos_c[-1])
        return sample_scale, m0_scale, m1_scale, mt_scale

    def build_step_coefficients(self) -> npt.NDArray[np.float32]:
        """Pre-compute all UniPC step coefficients as a dense numpy array.

        Returns array of shape ``[num_steps, 9]`` with columns
        ``[sigma, corrected_input_scale, corrector_sample_scale,
        corrector_m0_scale, corrector_m1_scale, corrector_mt_scale,
        predictor_sample_scale, predictor_m0_scale, predictor_m1_scale]``.

        Must be called AFTER ``set_timesteps()``.
        """
        if self.sigmas is None or self.timesteps is None:
            raise ValueError(
                "Scheduler must be initialized with set_timesteps()."
            )

        num_steps = len(self.timesteps)
        lower_order_nums = 0
        previous_predictor_order = 1
        rows: list[npt.NDArray[np.float32]] = []

        for step_index in range(num_steps):
            corrector_order = previous_predictor_order if step_index > 0 else 0
            corrected_input_scale = 1.0 if corrector_order == 0 else 0.0

            if self.lower_order_final:
                candidate_order = min(self.solver_order, num_steps - step_index)
            else:
                candidate_order = self.solver_order
            predictor_order = min(candidate_order, lower_order_nums + 1)

            p_sample, p_m0, p_m1 = self._predictor_coefficients(
                step_index, predictor_order
            )

            if corrector_order > 0:
                c_sample, c_m0, c_m1, c_mt = self._corrector_coefficients(
                    step_index, corrector_order
                )
            else:
                c_sample, c_m0, c_m1, c_mt = 0.0, 0.0, 0.0, 0.0

            rows.append(
                np.array(
                    [
                        float(self.sigmas[step_index]),
                        corrected_input_scale,
                        c_sample,
                        c_m0,
                        c_m1,
                        c_mt,
                        p_sample,
                        p_m0,
                        p_m1,
                    ],
                    dtype=np.float32,
                )
            )

            previous_predictor_order = predictor_order
            if lower_order_nums < self.solver_order:
                lower_order_nums += 1

        return np.stack(rows)
