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

"""Caching types and utilities for diffusion pipelines.

This module merges two formerly separate modules:
- cache_mixin.py: pipeline-level state and FBCache conditional execution
- denoising_cache.py: buffer-based TaylorSeerCache for executor-style pipelines

Settings, resolved config, and architecture defaults live in
:mod:`max.pipelines.diffusion.config`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, Graph, TensorType, TensorValue

from .config import DenoisingCacheConfig
from .taylorseer import TaylorSeer, _TaylorPredictModule, _TaylorUpdateModule

__all__ = [
    "DenoisingCacheState",
    "TaylorSeerBufferState",
    "TaylorSeerCache",
    "fbcache_conditional_execution",
]


@dataclass
class DenoisingCacheState:
    """Per-request mutable cache state for a single denoising stream.

    One instance per stream. Created fresh per execute() call.
    """

    # FBCache state
    prev_residual: Tensor | None = None
    prev_output: Tensor | None = None

    # TaylorSeer state
    taylor_factor_0: Tensor | None = None
    taylor_factor_1: Tensor | None = None
    taylor_factor_2: Tensor | None = None
    taylor_last_compute_step: int | None = None


def fbcache_conditional_execution(
    first_block_residual: Tensor,
    prev_residual: Tensor,
    prev_output: Tensor,
    residual_threshold: Tensor,
    run_remaining_blocks: Callable[..., Tensor],
    run_remaining_kwargs: dict[str, Any],
    run_postamble: Callable[[Tensor, Tensor], Tensor],
    temb: Tensor,
    output_types: list[TensorType],
) -> tuple[Tensor, Tensor]:
    """Handle FBCache F.cond branching pattern shared across DiT models.

    The caller provides atomic DiT methods:

    - *run_remaining_blocks*: runs blocks 1..N + single-stream blocks,
      returns pre-tail hidden states.
    - *run_postamble*: applies final norm + projection.

    The ``residual_threshold`` is a scalar Tensor (float32, shape=[]) passed
    as a graph input so it can be changed at runtime without recompilation.

    Returns:
        (first_block_residual, output) tensors.
    """
    use_fbcache = _can_use_fbcache(
        first_block_residual, prev_residual, residual_threshold
    )

    def then_fn(
        _prev_output: Tensor = prev_output,
        _first_block_residual: Tensor = first_block_residual,
    ) -> tuple[TensorValue, TensorValue]:
        return (
            TensorValue(_first_block_residual),
            TensorValue(_prev_output),
        )

    def else_fn(
        _fbr: Tensor = first_block_residual,
    ) -> tuple[TensorValue, TensorValue]:
        hidden_states = run_remaining_blocks(**run_remaining_kwargs)
        out = run_postamble(hidden_states, temb)
        return (TensorValue(_fbr), TensorValue(out))

    result = F.cond(
        use_fbcache,
        output_types,
        then_fn,
        else_fn,
    )
    return (result[0], result[1])


def _can_use_fbcache(
    intermediate_residual: Tensor,
    prev_intermediate_residual: Tensor | None,
    residual_threshold: Tensor,
) -> Tensor:
    """Return whether previous residual cache is reusable (RDT check).

    Args:
        intermediate_residual: First-block residual for the current step.
        prev_intermediate_residual: First-block residual from the previous step.
        residual_threshold: Scalar Tensor (float32, shape=[]) with the
            relative difference threshold.  Passed as a graph input so it
            can be changed at runtime without recompilation.
    """
    dev = intermediate_residual.device
    if (
        prev_intermediate_residual is None
        or intermediate_residual.shape != prev_intermediate_residual.shape
    ):
        return F.constant(False, DType.bool, device=dev)

    mean_diff_rows = F.mean(
        F.abs(intermediate_residual - prev_intermediate_residual), axis=-1
    )
    mean_prev_rows = F.mean(F.abs(prev_intermediate_residual), axis=-1)
    mean_diff = F.mean(mean_diff_rows, axis=None)
    mean_prev = F.mean(mean_prev_rows, axis=None)
    eps = 1e-9
    relative_diff = mean_diff / (mean_prev + eps)
    rdt = residual_threshold.cast(relative_diff.dtype)
    pred = relative_diff < rdt
    return F.squeeze(pred, 0)


# ---------------------------------------------------------------------------
# Buffer-based TaylorSeer cache for executor-style diffusion pipelines
# ---------------------------------------------------------------------------


@dataclass
class TaylorSeerBufferState:
    """Per-request mutable TaylorSeer state using Buffer objects.

    Allocated fresh for each denoising request via
    :meth:`TaylorSeerCache.create_state`.
    """

    factor_0: Buffer
    """Cached 0th-order factor (function value)."""

    factor_1: Buffer
    """Cached 1st-order factor (first derivative approximation)."""

    factor_2: Buffer
    """Cached 2nd-order factor (second derivative approximation)."""

    last_compute_step: int | None = None
    """Step index of the last full transformer computation."""


class TaylorSeerCache:
    """High-level TaylorSeer for executor pipelines (Buffer-based).

    Compiles predict and update graphs through the executor's shared
    :class:`InferenceSession` at construction time.  All runtime methods
    accept and return :class:`Buffer` objects, matching the executor's
    driver-level API.

    Args:
        config: Resolved denoising cache configuration (must have
            ``taylorseer=True``).
        dtype: Model compute dtype (e.g. ``DType.bfloat16``).
        device: Target device for graph execution.
        session: The executor's shared inference session.
    """

    def __init__(
        self,
        config: DenoisingCacheConfig,
        dtype: DType,
        device: Device,
        session: InferenceSession,
    ) -> None:
        self._dtype = dtype
        self._device = device

        self._cache_interval: int = config.taylorseer_cache_interval
        self._warmup_steps: int = config.taylorseer_warmup_steps
        self._max_order: int = config.taylorseer_max_order

        # Pre-allocate max_order scalar on device.
        self._max_order_buf: Buffer = Buffer.from_dlpack(
            np.array([self._max_order], dtype=np.int32)
        ).to(device)

        # Compile predict graph.
        device_ref = DeviceRef.from_device(device)
        predict_module = _TaylorPredictModule(dtype, device_ref)
        with Graph(
            "taylor_predict", input_types=predict_module.input_types()
        ) as predict_graph:
            predict_output = predict_module(
                *(value.tensor for value in predict_graph.inputs)
            )
            predict_graph.output(predict_output)
        self._predict_model: Model = session.load(predict_graph)

        # Compile update graph.
        update_module = _TaylorUpdateModule(dtype, device_ref)
        with Graph(
            "taylor_update", input_types=update_module.input_types()
        ) as update_graph:
            update_outputs = update_module(
                *(value.tensor for value in update_graph.inputs)
            )
            update_graph.output(*update_outputs)
        self._update_model: Model = session.load(update_graph)

    def create_state(
        self, batch_size: int, seq_len: int, output_dim: int
    ) -> TaylorSeerBufferState:
        """Allocate fresh per-request TaylorSeer state buffers.

        Args:
            batch_size: Batch dimension.
            seq_len: Sequence length (packed latent tokens).
            output_dim: Channel dimension of noise_pred.

        Returns:
            A new :class:`TaylorSeerBufferState` with zero-initialized
            factor buffers on the target device.
        """
        shape = (batch_size, seq_len, output_dim)
        return TaylorSeerBufferState(
            factor_0=Buffer.zeros(shape, self._dtype, device=self._device),
            factor_1=Buffer.zeros(shape, self._dtype, device=self._device),
            factor_2=Buffer.zeros(shape, self._dtype, device=self._device),
        )

    def should_skip(self, step: int) -> bool:
        """Return True when the full transformer pass can be skipped."""
        return TaylorSeer.should_skip(
            step, self._warmup_steps, self._cache_interval
        )

    def predict(self, state: TaylorSeerBufferState, step: int) -> Buffer:
        """Predict noise_pred from cached Taylor factors.

        Args:
            state: Current per-request TaylorSeer state.
            step: Current denoising step index.

        Returns:
            Predicted noise_pred buffer, shape ``(B, seq, C)``.
        """
        delta = (
            float(step - state.last_compute_step)
            if state.last_compute_step is not None
            else 1.0
        )
        step_offset = Buffer.from_dlpack(
            np.array([delta], dtype=np.float32)
        ).to(self._device)

        result = self._predict_model.execute(
            state.factor_0,
            state.factor_1,
            state.factor_2,
            step_offset,
            self._max_order_buf,
        )
        return result[0] if isinstance(result, (list, tuple)) else result

    def update(
        self,
        state: TaylorSeerBufferState,
        noise_pred: Buffer,
        step: int,
    ) -> None:
        """Update Taylor factors from a full transformer computation.

        Mutates *state* in-place with new factor values.

        Args:
            state: Current per-request TaylorSeer state.
            noise_pred: Fresh noise_pred from the transformer, shape
                ``(B, seq, C)``.
            step: Current denoising step index.
        """
        delta = (
            float(step - state.last_compute_step)
            if state.last_compute_step is not None
            else 1.0
        )
        delta_step = Buffer.from_dlpack(np.array([delta], dtype=np.float32)).to(
            self._device
        )

        results = self._update_model.execute(
            noise_pred,
            state.factor_0,
            state.factor_1,
            delta_step,
            self._max_order_buf,
        )
        state.factor_0 = results[0]
        state.factor_1 = results[1]
        state.factor_2 = results[2]
        state.last_compute_step = step
