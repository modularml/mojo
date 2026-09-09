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
"""Checkpoint-to-MAX weight conversion for MiniMax Music 3.

Every rewrite here happens once at load time rather than in the graph:

* **Weight-norm folding.** The vocoder stores ``weight_g``/``weight_v`` pairs.
  Recomputing ``g * v / ||v||`` on every call is what a weight-normalized layer
  does, but that factorization exists only to help training; folding it offline
  gives identical numerics at no per-call cost.
* **Layout permutes.** The convolutions run channels-last, which wants the kernel
  axis first rather than last.
* **Sub-pixel splits.** Each transposed convolution becomes two phase-major
  matrices, per the rewrite in ``layers/conv.py``.
* **Broadcast reshapes.** ``Snake1d``'s ``alpha`` is stored channel-first as
  ``(1, C, 1)`` and must broadcast over a channels-last activation.
* **Vocabulary slicing.** The global model's 200000-row output head is cut to
  the 16385 rows generation can emit and its input table is dropped entirely,
  which is 3.1 GiB the device does not have to hold.
* **A table moved between checkpoints.** The audio rows of that dropped input
  table are the one part of it still needed, and they are read by the depth
  stage, so they are loaded as one of the depth decoder's own weights.
"""

from __future__ import annotations

import numpy as np
from max.driver import CPU, Buffer, DLPackArray
from max.dtype import DType
from max.experimental.tensor import Tensor
from max.graph.weights import WeightData, Weights
from max.pipelines.lib.bfloat16_utils import float32_array_to_buffer

from .model_config import LanguageModelConfig

# Transposed convolutions in the vocoder, which become sub-pixel matmul pairs.
_TRANSPOSED_CONVS = ("conv_t1",)


def _conv_to_max_layout(arr: np.ndarray) -> np.ndarray:
    """Permute a PyTorch convolution kernel from ``(out, in, k)`` to ``(k, in, out)``.

    That is the layout the port's convolutions read, one matmul per tap: see
    :mod:`..layers.conv`.
    """
    return np.ascontiguousarray(arr.transpose(2, 1, 0))


def _split_subpixel(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a transposed kernel into the two phase-major halves.

    Args:
        arr: PyTorch ``ConvTranspose1d`` kernel, ``(in, out, 2 * stride)``.

    Returns:
        The ``[0, stride)`` and ``[stride, 2 * stride)`` tap sets, each folded to
        ``(in, stride * out)`` so a single matmul emits every output phase.
    """
    in_channels, out_channels, kernel = arr.shape
    stride = kernel // 2
    phase = arr.transpose(0, 2, 1)  # (in, tap, out)
    halves = (phase[:, :stride, :], phase[:, stride:, :])
    return tuple(  # type: ignore[return-value]
        np.ascontiguousarray(h).reshape(in_channels, stride * out_channels)
        for h in halves
    )


def _fold_weight_norm(g: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Reconstruct a weight-normalized kernel as ``g * v / ||v||``.

    ``weight_norm(dim=0)`` reduces every axis but the first, which for a 1D
    kernel means axes ``(1, 2)`` regardless of whether axis 0 is output channels
    (convolution) or input channels (transposed convolution).

    Args:
        g: Magnitude tensor of shape ``(axis0, 1, 1)``.
        v: Direction tensor of shape ``(axis0, other, k)``.

    Returns:
        The folded kernel in ``v``'s layout and dtype.
    """
    # Accumulate the norm in float64: summing 768*7 squared float32 values
    # loses a few bits to ordering, and this runs once per load.
    v64 = v.astype(np.float64)
    norm = np.sqrt(np.sum(v64 * v64, axis=(1, 2), keepdims=True))
    return (g.astype(np.float64) * v64 / norm).astype(v.dtype)


def convert_vocoder_state(weights: Weights) -> dict[str, np.ndarray]:
    """Convert vocoder checkpoint weights into MAX weight arrays.

    Args:
        weights: The opened checkpoint.

    Returns:
        Float32 arrays keyed to match the module's attribute paths.

    Raises:
        KeyError: If a ``weight_g`` has no matching ``weight_v``.
    """
    state = {
        name: np.from_dlpack(value.data().astype(DType.float32))
        for name, value in weights.items()
    }
    out: dict[str, np.ndarray] = {}
    for name, arr in state.items():
        if name.endswith(".weight_v"):
            continue
        if name.endswith(".weight_g"):
            base = name[: -len(".weight_g")]
            folded = _fold_weight_norm(arr, state[f"{base}.weight_v"])
            if base.rsplit(".", 1)[-1] in _TRANSPOSED_CONVS:
                lo, hi = _split_subpixel(folded)
                out[f"{base}.weight_lo"] = lo
                out[f"{base}.weight_hi"] = hi
            else:
                out[f"{base}.weight"] = _conv_to_max_layout(folded)
        elif name.endswith(".alpha"):
            out[name] = np.ascontiguousarray(arr.reshape(1, 1, -1))
        elif arr.ndim == 3:
            # dec_in_proj is the only convolution stored without weight norm.
            out[name] = _conv_to_max_layout(arr)
        else:
            out[name] = np.ascontiguousarray(arr)
    return _as_float32(out)


def convert_condition_encoder_state(
    weights: Weights,
) -> dict[str, np.ndarray]:
    """Convert condition-encoder checkpoint weights into MAX weight arrays.

    Only the projection kernel needs rewriting, into the channels-last layout the
    port's convolutions use. The layer-mixing logits and scale are used as
    stored, with the softmax left in the graph so the mapping to the checkpoint
    stays one to one.

    Args:
        weights: The opened checkpoint.

    Returns:
        Float32 arrays keyed to match the module's attribute paths.
    """
    state = {
        name: np.from_dlpack(value.data().astype(DType.float32))
        for name, value in weights.items()
    }
    return _as_float32(
        {
            name: _conv_to_max_layout(arr) if arr.ndim == 3 else arr
            for name, arr in state.items()
        }
    )


def convert_transformer_state(
    weights: Weights, dtype: DType
) -> dict[str, DLPackArray | WeightData]:
    """Convert flow-matching DiT checkpoint weights into MAX weight data.

    The only rewriting is to the two 1-tap convolutions, which the port runs as
    the matmuls they are: their ``(out, in, 1)`` kernels lose the trailing axis to
    become ``Linear`` weights. That is a metadata-only reinterpretation, so it
    goes through a buffer view rather than a copy.

    Args:
        weights: The opened checkpoint, typically sharded safetensors.
        dtype: Target dtype, ``bfloat16`` for this component.

    Returns:
        A state dict keyed to match the module's attribute paths.
    """
    state: dict[str, DLPackArray | WeightData] = {}
    for name, value in weights.items():
        data = value.data()
        if data.dtype != dtype:
            data = data.astype(dtype)
        if name.endswith(("preprocess_conv.weight", "postprocess_conv.weight")):
            state[name] = data.to_buffer().view(
                data.dtype, tuple(int(d) for d in data.shape[:2])
            )
        else:
            state[name] = data
    return state


def convert_depth_decoder_state(
    weights: Weights,
    language_model: Weights,
    config: LanguageModelConfig,
    dtype: DType,
) -> dict[str, DLPackArray | WeightData]:
    """Convert RVQ depth-decoder checkpoint weights into MAX weight data.

    This checkpoint needs no rewriting -- every key already matches the module's
    attribute path, and its embedding tables are stored the way
    :class:`~max.experimental.nn.Embedding` wants them -- but it is one table
    short of what the module needs. A frame's semantic code is embedded by the
    *global model's* table, and this stage is the only place that table's audio
    rows are read, so they are sliced out of the checkpoint and joined here (see
    :func:`audio_code_embeddings`). The dtype is imposed on both, so that a
    float32 gate can run against a bfloat16 checkpoint.

    Args:
        weights: The opened depth-decoder checkpoint.
        language_model: The opened language-model checkpoint, read only for the
            semantic codes' embedding rows.
        config: Supplies which rows those are.
        dtype: Target dtype.

    Returns:
        A state dict keyed to match the module's attribute paths.
    """
    state: dict[str, DLPackArray | WeightData] = {
        "semantic_embeddings.weight": audio_code_embeddings(
            language_model, config, dtype
        )
    }
    for name, value in weights.items():
        data = value.data()
        state[name] = data if data.dtype == dtype else data.astype(dtype)
    return state


def _row_slice(data: WeightData, rows: np.ndarray, dtype: DType) -> Buffer:
    """Gather rows of a 2-D weight, converting only what was gathered.

    Slicing before converting is the whole point: the callers want a few hundred
    rows of a 200000-row table, and converting first would materialize 3.3 GiB to
    discard almost all of it. bfloat16 has no numpy dtype, but a gather moves bit
    patterns and does not care what they mean, so the source is read through a
    uint16 view and only the result is widened.
    """
    if data.dtype != DType.bfloat16:
        raise ValueError(
            f"expected a bfloat16 checkpoint weight, got {data.dtype}"
        )
    gathered = np.ascontiguousarray(
        data.to_buffer().view(DType.uint16).to_numpy()[rows]
    )
    if dtype == DType.bfloat16:
        return Buffer.from_numpy(gathered).view(DType.bfloat16)
    widened = (gathered.astype(np.uint32) << 16).view(np.float32)
    return float32_array_to_buffer(widened, dtype=dtype, device=CPU())


def _emittable_rows(config: LanguageModelConfig) -> np.ndarray:
    """The vocabulary rows generation can reach: the codes, then the end token."""
    return np.append(
        np.arange(
            config.audio_code_offset,
            config.audio_code_offset + config.semantic_vocab_size,
        ),
        config.audio_end_token_id,
    )


def convert_language_model_state(
    weights: Weights, config: LanguageModelConfig, dtype: DType
) -> dict[str, DLPackArray | WeightData]:
    """Convert Qwen3 checkpoint weights into MAX weight data.

    Two rewrites are about the 200000-row vocabulary. The output head is cut to
    the 16385 rows the model can emit, and the input embedding table is dropped
    entirely: the port feeds embeddings rather than token ids, so the only rows it
    needs are the audio ones, and those go to the depth stage instead (see
    :func:`audio_code_embeddings`). Together that is 3.1 GiB the device does not
    have to hold.

    The third is the per-head QK norms, whose scales are plain tensors on the
    attention layer rather than child modules, so ``q_norm.weight`` becomes
    ``q_norm_weight``. The projections need no rename: ``QKVLinear`` omits its own
    attribute name from its children's paths, so ``q_proj`` stays ``q_proj``.

    Args:
        weights: The opened checkpoint.
        config: Supplies the row contract.
        dtype: Target dtype.

    Returns:
        A state dict keyed to match the module's attribute paths.
    """
    state: dict[str, DLPackArray | WeightData] = {}
    for name, value in weights.items():
        if name == "model.embed_tokens.weight":
            continue
        if name == "lm_head.weight":
            state["head.weight"] = _row_slice(
                value.data(), _emittable_rows(config), dtype
            )
            continue
        data = value.data()
        key = name.removeprefix("model.")
        for norm in ("q_norm", "k_norm"):
            key = key.replace(f".{norm}.weight", f".{norm}_weight")
        state[key] = data if data.dtype == dtype else data.astype(dtype)
    return state


def audio_code_embeddings(
    weights: Weights, config: LanguageModelConfig, dtype: DType
) -> Buffer:
    """Extract the semantic codes' rows of the language model's input embedding.

    A frame's semantic code is embedded by the *language model's* table, not the
    depth decoder's, which is the one detail that would otherwise force the whole
    200000-row table onto the device. Only ``semantic_vocab_size`` rows of it are
    ever addressed, so they travel with the depth stage that needs them.

    Args:
        weights: The opened language-model checkpoint.
        config: Supplies the offset and count.
        dtype: Target dtype.

    Returns:
        ``(semantic_vocab_size, hidden_size)`` embeddings.
    """
    rows = np.arange(
        config.audio_code_offset,
        config.audio_code_offset + config.semantic_vocab_size,
    )
    return _row_slice(weights["model.embed_tokens.weight"].data(), rows, dtype)


def embed_text(weights: Weights, token_ids: np.ndarray, dtype: DType) -> Tensor:
    """Look up the prompt's embeddings on the host, once.

    The prompt is the only place the full vocabulary is addressed, and it is
    addressed exactly once per request, so a host-side gather of a few hundred
    rows replaces a 1.6 GiB resident table.

    Args:
        weights: The opened language-model checkpoint.
        token_ids: ``(batch, prompt_length)`` ids.
        dtype: Target dtype.

    Returns:
        ``(batch * prompt_length, hidden_size)``, the sequences laid end to end
        as ragged attention wants them.
    """
    return Tensor(
        storage=_row_slice(
            weights["model.embed_tokens.weight"].data(),
            np.asarray(token_ids).reshape(-1),
            dtype,
        )
    )


def _as_float32(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name: np.ascontiguousarray(arr.astype(np.float32))
        for name, arr in state.items()
    }
