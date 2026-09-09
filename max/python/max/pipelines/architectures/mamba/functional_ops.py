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
"""Non-legacy functional ops for Mamba using F.custom() with custom_extensions.

These wrappers mirror the legacy ops.custom() calls in causal_conv1d.py,
selective_scan.py, and fused_norm.py, but use the non-legacy F.custom() API
with custom_extensions for loading state_space.mojoc.
"""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path

import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import Dim, TensorType

logger = logging.getLogger("max.pipelines.mamba")

_MODULAR_MOJO_MAX_IMPORT_PATH = "MODULAR_MOJO_MAX_IMPORT_PATH"


@functools.cache
def _get_state_space_paths() -> tuple[Path, ...]:
    """Get paths to state_space.mojoc kernel libraries.

    Reads the Mojo package locations and finds state_space.mojoc
    files. Results are cached since paths don't change during a session.
    """
    import_path_env = os.environ.get(_MODULAR_MOJO_MAX_IMPORT_PATH, "")
    if not import_path_env:
        site_packages = Path(__file__).resolve().parents[4]
        wheel_layout = site_packages / "modular"
        conda_layout = site_packages.parent.parent.parent
        for root in (wheel_layout, conda_layout):
            mojo_lib = root / "lib" / "mojo"
            if mojo_lib.is_dir():
                import_path_env = str(mojo_lib)
                break
    if not import_path_env:
        logger.warning(
            "MODULAR_MOJO_MAX_IMPORT_PATH not set for functional_ops"
        )
        return ()

    paths: list[Path] = []
    for entry in import_path_env.split(","):
        if not entry.strip():
            continue
        entry_path = Path(entry.strip())
        if not entry_path.is_absolute():
            resolved = Path.cwd() / entry_path
            if not resolved.exists():
                resolved = entry_path
            entry_path = resolved
        if not entry_path.exists():
            continue
        if entry_path.suffix == ".mojoc":
            if "state_space" in entry_path.name:
                paths.append(entry_path.resolve())
            continue
        if entry_path.is_dir():
            for mojoc in entry_path.rglob("*.mojoc"):
                if "state_space" in mojoc.name and (
                    mojoc.is_file() or mojoc.is_symlink()
                ):
                    paths.append(mojoc.resolve())
    logger.info(f"functional_ops found {len(paths)} state_space paths: {paths}")
    return tuple(paths)


def _normalize_activation(activation: str) -> str:
    """Normalize activation name for conv1d kernels."""
    act = activation.lower() if activation else "none"
    if act == "swish":
        act = "silu"
    return act if act in ("none", "silu") else "none"


# ---------------------------------------------------------------------------
# Causal Conv1D ops
# ---------------------------------------------------------------------------


def causal_conv1d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    activation: str = "silu",
    custom_extensions: tuple[Path, ...] | None = None,
) -> Tensor:
    """Causal 1D convolution (prefill, full sequence).

    Args:
        x: Input tensor of shape (batch, channels, seqlen).
        weight: Weight tensor of shape (channels, width).
        bias: Optional bias tensor of shape (channels,).
        activation: Activation function ("none" or "silu").
        custom_extensions: Paths to kernel libraries.

    Returns:
        Output tensor of shape (batch, channels, seqlen).
    """
    if custom_extensions is None:
        custom_extensions = _get_state_space_paths()

    activation_param = _normalize_activation(activation)

    weight_cast = weight.cast(x.dtype)

    if bias is None:
        bias_tensor = F.constant(0.0, dtype=x.dtype, device=x.device)
        bias_tensor = bias_tensor.broadcast_to([x.shape[1]])
    else:
        bias_tensor = bias.cast(x.dtype)

    result = F.custom(
        "causal_conv1d",
        x.device,
        [x, weight_cast, bias_tensor],
        [TensorType(dtype=x.dtype, shape=x.shape, device=x.device)],
        parameters={"activation": activation_param},
        custom_extensions=custom_extensions,
    )
    return result[0]


def causal_conv1d_update(
    x: Tensor,
    conv_state: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    activation: str = "silu",
    custom_extensions: tuple[Path, ...] | None = None,
) -> tuple[Tensor, Tensor]:
    """Causal 1D convolution update (step, single token).

    Uses the optimized Mojo causal_conv1d_update kernel. The op accepts
    the previous conv_state as an input and produces the updated state as
    a separate output (functional/pure semantics, no in-place mutation).

    Args:
        x: Input tensor of shape (batch, channels, 1).
        conv_state: Conv state of shape (batch, channels, width).
        weight: Weight tensor of shape (channels, width).
        bias: Optional bias tensor of shape (channels,).
        activation: Activation function ("none" or "silu").
        custom_extensions: Paths to kernel libraries.

    Returns:
        Tuple of (output, updated_conv_state).
        output: (batch, channels, 1)
        updated_conv_state: (batch, channels, width)
    """
    if custom_extensions is None:
        custom_extensions = _get_state_space_paths()

    activation_param = _normalize_activation(activation)

    weight_cast = weight.cast(x.dtype)
    conv_state_cast = conv_state.cast(x.dtype)

    if bias is None:
        bias_tensor = F.constant(0.0, dtype=x.dtype, device=x.device)
        bias_tensor = bias_tensor.broadcast_to([x.shape[1]])
    else:
        bias_tensor = bias.cast(x.dtype)

    output_type = TensorType(dtype=x.dtype, shape=x.shape, device=x.device)
    conv_state_out_type = TensorType(
        dtype=x.dtype, shape=conv_state.shape, device=x.device
    )

    results = F.custom(
        "causal_conv1d_update",
        x.device,
        [x, conv_state_cast, weight_cast, bias_tensor],
        [output_type, conv_state_out_type],
        parameters={"activation": activation_param},
        custom_extensions=custom_extensions,
    )
    return results[0], results[1]


# ---------------------------------------------------------------------------
# Selective Scan ops
# ---------------------------------------------------------------------------


def selective_scan_fwd(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor | None = None,
    z: Tensor | None = None,
    delta_bias: Tensor | None = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
    custom_extensions: tuple[Path, ...] | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """Selective scan forward pass (prefill, full sequence).

    Args:
        u: Input tensor of shape (batch, dim, seqlen).
        delta: Time step tensor of shape (batch, dim, seqlen).
        A: State transition matrix of shape (dim, dstate).
        B: Input projection of shape (batch, n_groups, dstate, seqlen).
        C: Output projection of shape (batch, n_groups, dstate, seqlen).
        D: Optional skip connection of shape (dim,).
        z: Optional gate tensor of shape (batch, dim, seqlen).
        delta_bias: Optional delta bias of shape (dim,).
        delta_softplus: Whether to apply softplus to delta.
        return_last_state: Whether to return the last SSM state.
        custom_extensions: Paths to kernel libraries.

    Returns:
        If return_last_state is False: output tensor (batch, dim, seqlen).
        If return_last_state is True: (output, last_state) where
            last_state is (batch, dim, dstate).
    """
    if custom_extensions is None:
        custom_extensions = _get_state_space_paths()

    device = u.device
    batch_dim = u.shape[0]
    dim_dim = u.shape[1]
    seqlen = u.shape[2]
    dstate = A.shape[1]

    chunk_size = 2048
    n_chunks = (seqlen + Dim(chunk_size) - Dim(1)) // Dim(chunk_size)

    has_z = z is not None
    has_D = D is not None
    has_delta_bias = delta_bias is not None
    use_minimal_kernel = not has_D and not has_z and not has_delta_bias

    output_type = TensorType(
        dtype=u.dtype, shape=[batch_dim, dim_dim, seqlen], device=device
    )
    x_checkpoint_type = TensorType(
        dtype=u.dtype,
        shape=[batch_dim, dim_dim, n_chunks, Dim(2) * dstate],
        device=device,
    )

    if use_minimal_kernel:
        results = F.custom(
            "selective_scan_fwd_minimal",
            device,
            [u, delta, A, B, C],
            [output_type, x_checkpoint_type],
            parameters={"delta_softplus": delta_softplus},
            custom_extensions=custom_extensions,
        )
        output = results[0]
        x_checkpoint = results[1]
    else:
        out_z_type = TensorType(
            dtype=u.dtype,
            shape=[batch_dim, dim_dim, seqlen] if has_z else [0, 0, 0],
            device=device,
        )

        if D is None:
            D = F.constant(
                np.array([], dtype=np.float32), dtype=u.dtype, device=device
            )
        if z is None:
            z = F.constant(
                np.zeros((0, 0, 0), dtype=np.float32),
                dtype=u.dtype,
                device=device,
            )
        if delta_bias is None:
            delta_bias = F.constant(
                np.array([], dtype=np.float32), dtype=u.dtype, device=device
            )

        results = F.custom(
            "selective_scan_fwd",
            device,
            [u, delta, A, B, C, D, z, delta_bias],
            [output_type, x_checkpoint_type, out_z_type],
            parameters={"delta_softplus": delta_softplus},
            custom_extensions=custom_extensions,
        )

        if has_z:
            output = results[2]  # gated output: output * silu(z)
        else:
            output = results[0]
        x_checkpoint = results[1]

    if return_last_state:
        last_state = _extract_last_state(
            x_checkpoint, batch_dim, dim_dim, dstate
        )
        return output, last_state

    return output


def _extract_last_state(
    checkpoint: Tensor,
    batch_dim: Dim,
    dim_dim: Dim,
    dstate: Dim,
) -> Tensor:
    """Extract the last SSM state from the checkpoint tensor.

    The checkpoint stores cum_a and cum_b interleaved per state element:
      [cum_a[0], cum_b[0], cum_a[1], cum_b[1], ...]
    i.e. cum_a at even indices (0, 2, 4, ...) and cum_b at odd indices
    (1, 3, 5, ...). The actual SSM state is cum_b.

    checkpoint shape: (batch, dim, n_chunks, 2*dstate)
    Returns: (batch, dim, dstate)
    """
    # Get the last chunk: checkpoint[:, :, -1, :] -> (batch, dim, 2*dstate)
    last_chunk = checkpoint[:, :, -1, :]
    # Reshape: (batch, dim, 2*dstate) -> (batch, dim, dstate, 2)
    # This deinterleaves: [a0, b0, a1, b1, ...] -> [[a0,b0], [a1,b1], ...]
    last_chunk = F.reshape(last_chunk, [batch_dim, dim_dim, dstate, 2])
    # Extract cum_b (index 1 along axis 3): (batch, dim, dstate)
    last_state = last_chunk[:, :, :, 1]
    return last_state


def selective_scan_update(
    state: Tensor,
    x: Tensor,
    dt: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor | None = None,
    z: Tensor | None = None,
    dt_bias: Tensor | None = None,
    dt_softplus: bool = False,
    custom_extensions: tuple[Path, ...] | None = None,
) -> tuple[Tensor, Tensor]:
    """Selective scan state update (step, single token).

    Args:
        state: SSM state of shape (batch, dim, dstate).
        x: Input of shape (batch, dim).
        dt: Delta/timestep of shape (batch, dim).
        A: State transition matrix of shape (dim, dstate).
        B: Input projection of shape (batch, n_groups, dstate).
        C: Output projection of shape (batch, n_groups, dstate).
        D: Optional skip connection of shape (dim,).
        z: Optional gate of shape (batch, dim).
        dt_bias: Optional delta bias of shape (dim,).
        dt_softplus: Whether to apply softplus to dt.
        custom_extensions: Paths to kernel libraries.

    Returns:
        Tuple of (updated_state, output) where:
            updated_state: (batch, dim, dstate)
            output: (batch, dim)
    """
    if custom_extensions is None:
        custom_extensions = _get_state_space_paths()

    device = state.device
    batch_dim = state.shape[0]
    dim_dim = state.shape[1]
    dstate_dim = state.shape[2]

    if D is None:
        D = F.constant(
            np.array([], dtype=np.float32).reshape(0),
            dtype=state.dtype,
            device=device,
        )
    if z is None:
        z = F.constant(
            np.array([], dtype=np.float32).reshape(0, 0),
            dtype=state.dtype,
            device=device,
        )
    if dt_bias is None:
        dt_bias = F.constant(
            np.array([], dtype=np.float32).reshape(0),
            dtype=state.dtype,
            device=device,
        )

    state_out_type = TensorType(
        dtype=state.dtype,
        shape=[batch_dim, dim_dim, dstate_dim],
        device=device,
    )
    output_type = TensorType(
        dtype=state.dtype,
        shape=[batch_dim, dim_dim],
        device=device,
    )

    results = F.custom(
        "selective_scan_update",
        device,
        [state, x, dt, A, B, C, D, z, dt_bias],
        [state_out_type, output_type],
        parameters={"delta_softplus": dt_softplus},
        custom_extensions=custom_extensions,
    )

    return results[0], results[1]


# ---------------------------------------------------------------------------
# Norm ops
# ---------------------------------------------------------------------------


def rms_norm_fused_residual(
    x: Tensor,
    residual: Tensor,
    weight: Tensor,
    eps: float,
    weight_offset: float = 0.0,
    multiply_before_cast: bool = False,
    custom_extensions: tuple[Path, ...] | None = None,
) -> tuple[Tensor, Tensor]:
    """Fused RMSNorm with residual addition: norm(x + residual).

    Args:
        x: Input tensor of shape (*, hidden).
        residual: Residual tensor of shape (*, hidden).
        weight: Norm weight of shape (hidden,).
        eps: Epsilon for numerical stability.
        weight_offset: Offset added to weights (0.0 for standard, 1.0 for Gemma).
        multiply_before_cast: Whether to multiply before casting.
        custom_extensions: Paths to kernel libraries.

    Returns:
        Tuple of (normalized, updated_residual).
    """
    if custom_extensions is None:
        custom_extensions = _get_state_space_paths()

    weight_cast = weight.cast(x.dtype)
    eps_const = F.constant(eps, dtype=DType.float32, device=CPU())
    weight_offset_const = F.constant(weight_offset, dtype=x.dtype, device=CPU())
    dropout_p_const = F.constant(0.0, dtype=x.dtype, device=CPU())
    seed_const = F.constant(0, dtype=DType.uint64, device=CPU())

    results = F.custom(
        "rms_norm_fused_residual",
        x.device,
        [
            x,
            residual,
            weight_cast,
            eps_const,
            weight_offset_const,
            dropout_p_const,
            seed_const,
        ],
        [x.type, x.type],
        parameters={"multiply_before_cast": multiply_before_cast},
        custom_extensions=custom_extensions,
    )

    return results[0], results[1]
