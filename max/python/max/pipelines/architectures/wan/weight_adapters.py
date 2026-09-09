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

"""Weight adapter for Comfy-Org ``fp8_scaled`` Wan 2.2 DiT checkpoints.

The Comfy-Org / Kijai ``fp8_scaled`` safetensors use the **native Wan-AI**
parameter naming (``blocks.N.self_attn.q``, ``ffn.0``, ``head.head``,
``modulation``), not the diffusers naming that :func:`model._remap_state_dict`
handles. Every linear weight is ``float8_e4m3fn`` with a scalar
``scale_weight`` and scalar ``scale_input`` (static per-tensor W8A8). This
adapter maps native names onto the MAX Wan DiT module names and emits the
``weight`` / ``bias`` / ``weight_scale`` / ``input_scale`` tensors each FP8
:class:`~max.nn.linear.Linear` expects.

Both the high-noise (``transformer``) and low-noise (``transformer_2``)
experts share this layout. Wan 2.2 T2V and I2V differ only in the
``patch_embedding`` input-channel count, which is read from the tensor shape.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.graph.buffer_utils import cast_dlpack_to
from max.graph.weights import Weights

logger = logging.getLogger("max.pipelines")

# Scale-convention toggles. MAX's ``quantize_static_scaled_float8`` (used by
# the static FP8 matmul path) defaults to ``scale_is_inverted=True``, i.e. it
# expects the input scale as ``1 / amax``. Comfy-Org stores ``scale_input`` as
# the raw dequant multiplier, so it is inverted here by default. The weight
# scale is consumed as a dequant multiplier and passed through unchanged.
# Both are overridable via env var so the convention can be pinned down
# empirically against the BF16 reference without a code change.
_INVERT_INPUT_SCALE = os.environ.get("WAN_FP8_INVERT_INPUT_SCALE", "1") != "0"
_INVERT_WEIGHT_SCALE = os.environ.get("WAN_FP8_INVERT_WEIGHT_SCALE", "0") != "0"

# Native (Comfy) → MAX (diffusers-based) module-name remaps.
_TOP_LEVEL_LINEARS = {
    "time_embedding.0": "condition_embedder.time_embedder.linear_1",
    "time_embedding.2": "condition_embedder.time_embedder.linear_2",
    "time_projection.1": "condition_embedder.time_proj",
    "text_embedding.0": "condition_embedder.text_embedder.linear_1",
    "text_embedding.2": "condition_embedder.text_embedder.linear_2",
    "head.head": "proj_out",
}

# Per-block linear remaps (native sub-name → MAX sub-name), relative to
# ``blocks.N.``.
_BLOCK_LINEARS = {
    "self_attn.q": "attn1.to_q",
    "self_attn.k": "attn1.to_k",
    "self_attn.v": "attn1.to_v",
    "self_attn.o": "attn1.to_out",
    "cross_attn.q": "attn2.to_q",
    "cross_attn.k": "attn2.to_k",
    "cross_attn.v": "attn2.to_v",
    "cross_attn.o": "attn2.to_out",
    "ffn.0": "ffn.proj",
    "ffn.2": "ffn.linear_out",
}

# Per-block non-linear (bf16) remaps.
_BLOCK_BF16 = {
    "self_attn.norm_q.weight": "attn1.norm_q.weight",
    "self_attn.norm_k.weight": "attn1.norm_k.weight",
    "cross_attn.norm_q.weight": "attn2.norm_q.weight",
    "cross_attn.norm_k.weight": "attn2.norm_k.weight",
    # Native ``norm3`` is the (affine) cross-attention LayerNorm, which is
    # ``norm2`` in the diffusers/MAX block layout.
    "norm3.weight": "norm2.weight",
    "norm3.bias": "norm2.bias",
}


def is_wan_fp8_checkpoint(weights: Weights) -> bool:
    """Return ``True`` if ``weights`` is a Comfy-Org ``fp8_scaled`` Wan DiT.

    Detected by the ``scaled_fp8`` marker tensor that Comfy-Org writes at the
    top level of every ``fp8_scaled`` safetensor.
    """
    return any(key == "scaled_fp8" for key, _ in weights.items())


def adapt_wan_fp8_weights(
    weights: Weights, target_dtype: DType = DType.bfloat16
) -> dict[str, Any]:
    """Map a native Comfy ``fp8_scaled`` Wan state dict onto MAX module names.

    Args:
        weights: Loaded Comfy-Org ``fp8_scaled`` safetensors.
        target_dtype: Working dtype for the non-FP8 tensors (norms, biases,
            modulation, patch embedding). Defaults to ``bfloat16``.

    Returns:
        A state dict keyed by MAX Wan DiT parameter names. FP8 linears emit
        ``<name>.weight`` (float8_e4m3fn), ``<name>.bias`` (bfloat16),
        ``<name>.weight_scale`` and ``<name>.input_scale`` (float32 scalars).
    """
    cpu = CPU()
    native = {key: value.data() for key, value in weights.items()}
    out: dict[str, Any] = {}

    def _buf(wd: Any) -> Any:
        return wd.to_buffer() if hasattr(wd, "to_buffer") else wd

    def _emit_cast(max_key: str, wd: Any, dtype: DType) -> None:
        out[max_key] = cast_dlpack_to(_buf(wd), wd.dtype, dtype, cpu)

    def _emit_scale(max_key: str, wd: Any, invert: bool) -> None:
        f32 = cast_dlpack_to(_buf(wd), wd.dtype, DType.float32, cpu)
        value = float(
            np.asarray(np.from_dlpack(f32), dtype=np.float32).reshape(-1)[0]
        )
        if invert:
            value = 1.0 / value
        # Linear's per-tensor weight_scale / input_scale are scalars (shape ()).
        out[max_key] = np.array(value, dtype=np.float32)

    def _emit_linear(native_prefix: str, max_prefix: str) -> None:
        _emit_cast(
            f"{max_prefix}.weight",
            native[f"{native_prefix}.weight"],
            DType.float8_e4m3fn,
        )
        bias_key = f"{native_prefix}.bias"
        if bias_key in native:
            _emit_cast(f"{max_prefix}.bias", native[bias_key], target_dtype)
        _emit_scale(
            f"{max_prefix}.weight_scale",
            native[f"{native_prefix}.scale_weight"],
            _INVERT_WEIGHT_SCALE,
        )
        _emit_scale(
            f"{max_prefix}.input_scale",
            native[f"{native_prefix}.scale_input"],
            _INVERT_INPUT_SCALE,
        )

    # -- Top level -----------------------------------------------------------
    # patch_embedding Conv3d: diffusers [F, C, D, H, W] -> MAX [D, H, W, C, F].
    pe_w = native["patch_embedding.weight"]
    pe_f32 = cast_dlpack_to(_buf(pe_w), pe_w.dtype, DType.float32, cpu)
    out["patch_embedding.weight"] = cast_dlpack_to(
        np.ascontiguousarray(np.from_dlpack(pe_f32).transpose(2, 3, 4, 1, 0)),
        DType.float32,
        target_dtype,
        cpu,
    )
    _emit_cast(
        "patch_embedding.bias", native["patch_embedding.bias"], target_dtype
    )

    for native_prefix, max_prefix in _TOP_LEVEL_LINEARS.items():
        _emit_linear(native_prefix, max_prefix)

    # head.modulation -> post-process scale_shift_table (1, 2, dim).
    _emit_cast("scale_shift_table", native["head.modulation"], target_dtype)

    # -- Blocks --------------------------------------------------------------
    block_indices = sorted(
        {int(key.split(".")[1]) for key in native if key.startswith("blocks.")}
    )
    for n in block_indices:
        bp = f"blocks.{n}"
        for sub_native, sub_max in _BLOCK_LINEARS.items():
            _emit_linear(f"{bp}.{sub_native}", f"{bp}.{sub_max}")
        for sub_native, sub_max in _BLOCK_BF16.items():
            _emit_cast(
                f"{bp}.{sub_max}", native[f"{bp}.{sub_native}"], target_dtype
            )
        # modulation -> per-block scale_shift_table (1, 6, dim).
        _emit_cast(
            f"{bp}.scale_shift_table",
            native[f"{bp}.modulation"],
            target_dtype,
        )

    logger.info(
        "Adapted Wan FP8 checkpoint: %d MAX params from %d blocks "
        "(invert_input=%s, invert_weight=%s).",
        len(out),
        len(block_indices),
        _INVERT_INPUT_SCALE,
        _INVERT_WEIGHT_SCALE,
    )
    return out
