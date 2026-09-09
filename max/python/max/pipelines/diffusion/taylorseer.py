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

"""Standalone TaylorSeer caching module for diffusion pipelines.

TaylorSeer uses Taylor series approximation to skip full transformer
passes during denoising.  On steps where the transformer is skipped,
it predicts the output from cached Taylor factors (function value and
derivatives).  On steps where the transformer runs, it updates those
factors via divided differences.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from max._core.driver import Device
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn.layer import Module

if TYPE_CHECKING:
    from .cache import DenoisingCacheState
    from .config import DenoisingCacheConfig


@dataclass
class TaylorSeerState:
    """Per-request mutable state for TaylorSeer caching.

    Allocated fresh for each denoising request via
    ``TaylorSeer.create_state()``.
    """

    factor_0: Tensor | None = None
    """Cached 0th-order factor (function value)."""

    factor_1: Tensor | None = None
    """Cached 1st-order factor (first derivative approximation)."""

    factor_2: Tensor | None = None
    """Cached 2nd-order factor (second derivative approximation)."""

    last_compute_step: int | None = None
    """Step index of the last full transformer computation."""


class _TaylorPredictModule(Module):
    """ModuleV2 wrapper for the Taylor series prediction graph."""

    def __init__(self, dtype: DType, device: DeviceRef) -> None:
        super().__init__()
        self.dtype = dtype
        self.device = device

    def input_types(self) -> tuple[TensorType, ...]:
        tensor_type = TensorType(
            self.dtype,
            shape=["batch", "seq", "channels"],
            device=self.device,
        )
        scalar_type = TensorType(DType.float32, shape=[1], device=self.device)
        order_type = TensorType(DType.int32, shape=[1], device=self.device)
        return (
            tensor_type,  # factor_0
            tensor_type,  # factor_1
            tensor_type,  # factor_2
            scalar_type,  # step_offset
            order_type,  # max_order
        )

    def __call__(
        self,
        factor_0: TensorValue,
        factor_1: TensorValue,
        factor_2: TensorValue,
        step_offset: TensorValue,
        max_order: TensorValue,
    ) -> TensorValue:
        """Taylor series prediction: f(t+dt) ~ f(t) + f'(t)*dt + f''(t)*dt^2/2."""
        offset = ops.cast(step_offset, factor_0.dtype)
        result = factor_0 + factor_1 * offset
        offset_sq_half = (
            offset
            * offset
            * ops.constant(0.5, factor_0.dtype, device=factor_0.device)
        )
        order2_term = factor_2 * offset_sq_half
        use_order2 = max_order >= ops.constant(
            2, DType.int32, device=max_order.device
        )
        use_order2_cast = ops.cast(
            ops.broadcast_to(use_order2, order2_term.shape),
            order2_term.dtype,
        )
        result = result + order2_term * use_order2_cast
        return result


class _TaylorUpdateModule(Module):
    """ModuleV2 wrapper for the Taylor factor update graph."""

    def __init__(self, dtype: DType, device: DeviceRef) -> None:
        super().__init__()
        self.dtype = dtype
        self.device = device

    def input_types(self) -> tuple[TensorType, ...]:
        tensor_type = TensorType(
            self.dtype,
            shape=["batch", "seq", "channels"],
            device=self.device,
        )
        scalar_type = TensorType(DType.float32, shape=[1], device=self.device)
        order_type = TensorType(DType.int32, shape=[1], device=self.device)
        return (
            tensor_type,  # new_output
            tensor_type,  # old_factor_0
            tensor_type,  # old_factor_1
            scalar_type,  # delta_step
            order_type,  # max_order
        )

    def __call__(
        self,
        new_output: TensorValue,
        old_factor_0: TensorValue,
        old_factor_1: TensorValue,
        delta_step: TensorValue,
        max_order: TensorValue,
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        """Compute Taylor factors via divided differences."""
        delta = ops.cast(delta_step, new_output.dtype)
        eps = ops.constant(1e-9, new_output.dtype, device=new_output.device)
        safe_delta = delta + eps

        new_factor_0 = new_output
        new_factor_1 = (new_output - old_factor_0) / safe_delta
        new_factor_2 = (new_factor_1 - old_factor_1) / safe_delta
        use_order2 = max_order >= ops.constant(
            2, DType.int32, device=max_order.device
        )
        use_order2_cast = ops.cast(
            ops.broadcast_to(use_order2, new_factor_2.shape),
            new_factor_2.dtype,
        )
        new_factor_2 = new_factor_2 * use_order2_cast

        return new_factor_0, new_factor_1, new_factor_2


class TaylorSeer:
    """Standalone TaylorSeer caching module.

    Compiles ``predict`` and ``update`` graphs at construction time and
    provides methods for the full TaylorSeer lifecycle: scheduling,
    prediction, factor updates, and state allocation.
    """

    def __init__(self, max_order: int, dtype: DType, device: Device) -> None:
        self.max_order = max_order
        self.dtype = dtype
        self.device = device

        self.max_order_tensor = Tensor(
            storage=Buffer.from_dlpack(
                np.array([max_order], dtype=np.int32)
            ).to(device)
        )

        device_ref = DeviceRef.from_device(device)
        self._session = InferenceSession([device])

        predict_module = _TaylorPredictModule(dtype, device_ref)
        with Graph(
            "taylor_predict", input_types=predict_module.input_types()
        ) as predict_graph:
            predict_output = predict_module(
                *(value.tensor for value in predict_graph.inputs)
            )
            predict_graph.output(predict_output)
        self._predict_model: Model = self._session.load(predict_graph)

        update_module = _TaylorUpdateModule(dtype, device_ref)
        with Graph(
            "taylor_update", input_types=update_module.input_types()
        ) as update_graph:
            update_outputs = update_module(
                *(value.tensor for value in update_graph.inputs)
            )
            update_graph.output(*update_outputs)
        self._update_model: Model = self._session.load(update_graph)

    def compiled_predict(
        self,
        factor_0: Tensor,
        factor_1: Tensor,
        factor_2: Tensor,
        step_offset: Tensor,
        max_order: Tensor,
    ) -> Tensor:
        """Run the compiled Taylor predict graph on eager tensors."""
        buffers = self._predict_model.execute(
            factor_0.driver_tensor,
            factor_1.driver_tensor,
            factor_2.driver_tensor,
            step_offset.driver_tensor,
            max_order.driver_tensor,
        )
        return Tensor.from_dlpack(buffers[0])

    def compiled_update(
        self,
        new_output: Tensor,
        old_factor_0: Tensor,
        old_factor_1: Tensor,
        delta_step: Tensor,
        max_order: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run the compiled Taylor update graph on eager tensors."""
        buffers = self._update_model.execute(
            new_output.driver_tensor,
            old_factor_0.driver_tensor,
            old_factor_1.driver_tensor,
            delta_step.driver_tensor,
            max_order.driver_tensor,
        )
        return (
            Tensor.from_dlpack(buffers[0]),
            Tensor.from_dlpack(buffers[1]),
            Tensor.from_dlpack(buffers[2]),
        )

    @staticmethod
    def should_skip(step: int, warmup_steps: int, cache_interval: int) -> bool:
        """Return True when the full transformer pass can be skipped at *step*."""
        if step < warmup_steps:
            return False
        return (step - warmup_steps - 1) % cache_interval != 0

    def create_state(
        self, batch_size: int, seq_len: int, output_dim: int
    ) -> TaylorSeerState:
        """Allocate fresh per-request TaylorSeer state tensors."""
        shape = (batch_size, seq_len, output_dim)

        def _zeros() -> Tensor:
            return Tensor(
                storage=Buffer.zeros(shape, self.dtype, device=self.device)
            )

        return TaylorSeerState(
            factor_0=_zeros(),
            factor_1=_zeros(),
            factor_2=_zeros(),
        )


def run_denoising_step(
    step: int,
    cache_state: DenoisingCacheState,
    cache_config: DenoisingCacheConfig,
    device: Device,
    compute_fn: Callable[[], tuple[Tensor, ...]],
    taylorseer: TaylorSeer | None = None,
) -> Tensor:
    """Execute one denoising step with caching logic.

    This is a standalone version of ``DiffusionPipeline.run_denoising_step``
    that does not require inheritance.  The caller provides a *compute_fn*
    callback that runs the transformer and returns the raw result tuple.

    Args:
        step: Current step index.
        cache_state: Per-request mutable cache state for this stream.
        cache_config: Cache configuration.
        device: Target device.
        compute_fn: Callable that runs the transformer and returns the
            result tuple.  The tuple format depends on the cache mode:
            ``(noise_pred,)`` for standard,
            ``(new_residual, noise_pred)`` for FBCache.
        taylorseer: Optional TaylorSeer instance.  Required when
            ``cache_config.taylorseer`` is True.

    Returns:
        noise_pred tensor for this step.
    """
    # 1. TaylorSeer scheduling decision
    skip_transformer = False
    if cache_config.taylorseer:
        assert taylorseer is not None
        skip_transformer = taylorseer.should_skip(
            step,
            cache_config.taylorseer_warmup_steps,
            cache_config.taylorseer_cache_interval,
        )

    # 2. Compute TaylorSeer step delta
    taylor_delta_tensor: Tensor | None = None
    if cache_config.taylorseer:
        assert taylorseer is not None
        assert cache_state.taylor_factor_0 is not None
        assert cache_state.taylor_factor_1 is not None
        delta = (
            float(step - cache_state.taylor_last_compute_step)
            if cache_state.taylor_last_compute_step is not None
            else 1.0
        )
        taylor_delta_tensor = Tensor(
            storage=Buffer.from_dlpack(np.array([delta], dtype=np.float32)).to(
                device
            )
        )

    # 3. Predict path (skip transformer)
    if cache_config.taylorseer and skip_transformer:
        assert taylorseer is not None
        assert cache_state.taylor_factor_0 is not None
        assert cache_state.taylor_factor_1 is not None
        assert cache_state.taylor_factor_2 is not None
        assert taylor_delta_tensor is not None
        return taylorseer.compiled_predict(
            cache_state.taylor_factor_0,
            cache_state.taylor_factor_1,
            cache_state.taylor_factor_2,
            taylor_delta_tensor,
            taylorseer.max_order_tensor,
        )

    # 4. Full compute path
    result = compute_fn()
    if cache_config.first_block_caching:
        new_residual, noise_pred = result
        cache_state.prev_residual = new_residual
        cache_state.prev_output = noise_pred
    else:
        noise_pred = result[0]

    # 5. TaylorSeer factor update
    if cache_config.taylorseer:
        assert taylorseer is not None
        assert cache_state.taylor_factor_0 is not None
        assert cache_state.taylor_factor_1 is not None
        assert taylor_delta_tensor is not None
        (
            cache_state.taylor_factor_0,
            cache_state.taylor_factor_1,
            cache_state.taylor_factor_2,
        ) = taylorseer.compiled_update(
            noise_pred,
            cache_state.taylor_factor_0,
            cache_state.taylor_factor_1,
            taylor_delta_tensor,
            taylorseer.max_order_tensor,
        )
        cache_state.taylor_last_compute_step = step

    return noise_pred
