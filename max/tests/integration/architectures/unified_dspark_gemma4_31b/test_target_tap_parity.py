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
"""Target tap parity for the 31B DSpark speculators port (off-by-one proof).

The RedHat ``gemma-4-31B-it-speculator.dspark`` checkpoint declares
``aux_hidden_state_layer_ids = [1, 17, 29, 47, 58]`` in the vLLM eagle
convention: aux id ``j`` is the raw residual stream (hidden + residual, no
norm) at the INPUT of target layer ``j``, i.e. the output of layer ``j - 1``
(vllm ``eagle3_utils.py`` / ``interfaces.py:1405-1416``). MAX's shared
``Gemma4TextModel`` SELECTED_LAYERS capture appends AFTER layer ``idx`` runs,
so MAX id ``k`` is the OUTPUT of layer ``k``. The port must therefore wire
``target_layer_ids = [j - 1 for j in aux_ids] = [0, 16, 28, 46, 57]``. An
off-by-one here passes every compile/serve check and silently craters
acceptance, so the shift is proven once here against the real
``google/gemma-4-31B-it`` weights.

Method (one GPU; the bazel pytest runner statically partitions the
``gpu-memory`` request between torch and MAX's DeviceContext, so the torch
GPU pass and the MAX pass each fit their partition and the fp32 oracle runs
on host RAM):

1. HF torch bf16 reference on GPU: forward pre-hooks dump the tensors
   entering decoder layers [1, 2, 17, 29, 47, 58] on a fixed prompt set,
   then the model is freed.
2. HF torch float32 oracle on CPU: the same dumps at float32, to attribute
   bf16 divergence to rounding rather than semantics (12B oracle-margin
   pattern, ``unified_dspark_gemma4_12b/test_dspark_gemma4_module.py``).
3. MAX ``Gemma4TextModel`` with SELECTED_LAYERS capturing layer outputs
   [0, 1, 16, 17, 28, 29, 46, 47, 57, 58] in one compiled graph, so the
   positive proof and the negative control share a single 60-layer compile.

Gates:

- positive: MAX capture ``j - 1`` matches the torch input of layer ``j``
  within ORACLE_MARGIN x the bf16 reference's own error vs the fp32 oracle,
  plus an absolute backstop;
- adjacency: MAX capture 1 also matches the torch input of layer 2, nailing
  the +1 relationship directly;
- negative control (assert-inverted, always on): MAX capture ``j`` — the
  naive id-for-id wiring of the checkpoint's aux ids — must FAIL both
  positive gates (inverted oracle margin, absolute backstop) and diverge
  from the torch input of layer ``j`` by at least 10x the matched pair's
  distance, proving this test fails on the off-by-one wiring.
"""

from __future__ import annotations

import gc
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from huggingface_hub import snapshot_download
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.graph.weights import load_weights
from max.nn.comm import Signals
from max.nn.kv_cache import MHAKVCacheParams, MultiKVCacheParams
from max.nn.transformer import ReturnHiddenStates
from max.pipelines.architectures.gemma4.gemma4 import Gemma4TextModel
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalScalingParams,
)
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
    Gemma4TextConfig,
)
from max.pipelines.architectures.gemma4.weight_adapters import (
    convert_safetensor_language_state_dict,
)
from max.pipelines.kv_cache import PagedKVCacheManager
from test_common.context_utils import create_text_context
from torch.utils.dlpack import from_dlpack
from transformers import AutoTokenizer
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForConditionalGeneration,
)

TARGET_REPO_ID = "google/gemma-4-31B-it"
MAX_DTYPE = DType.bfloat16

# From the RedHat checkpoint's speculators config (vLLM eagle convention);
# see evidence in the module docstring.
VLLM_AUX_LAYER_IDS = [1, 17, 29, 47, 58]
MAX_TARGET_LAYER_IDS = [j - 1 for j in VLLM_AUX_LAYER_IDS]
# Extra torch tap at layer 2 so MAX capture 1 can be shown to equal the
# input of layer 2 (and not the input of layer 1).
ADJACENCY_TORCH_LAYER = 2
TORCH_TAP_LAYERS = sorted(set(VLLM_AUX_LAYER_IDS) | {ADJACENCY_TORCH_LAYER})
# One graph captures both the claimed wiring and the off-by-one wiring.
MAX_CAPTURE_IDS = sorted(set(MAX_TARGET_LAYER_IDS) | set(VLLM_AUX_LAYER_IDS))

# 12B fp32-oracle margin pattern: MAX bf16 must be about as close to the
# fp32 truth as the torch bf16 reference's own rounding error.
ORACLE_MARGIN = 2.5
# Absolute backstop, sized to separate rounding from semantics on this
# fixed prompt set: matched taps measure 6e-6..4e-4 while the off-by-one
# wiring measures 5.7e-3..0.56 (adjacent residual streams stay correlated
# at depth, so mismatches at deep taps are small in absolute terms).
HS_COS_DIST_BACKSTOP = 2e-3
# The negative control must exceed this multiple of the correctly-paired
# distance, or the test cannot claim sensitivity (measured: >= 130x).
NEGATIVE_SEPARATION = 10.0

PROMPTS = [
    "The capital of France is Paris, and the capital of Japan is",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number,'
    ' iteratively."""\n    a, b = 0, 1\n    for _ in range(n):',
    "Attention mechanisms let a transformer weigh every token in the"
    " context against every other token, so the residual stream at each"
    " layer accumulates increasingly abstract features of the sequence."
    " Speculative decoding exploits the fact that a small draft model can"
    " often predict the next few tokens of a much larger model.",
]


_PreHook = Callable[
    [object, tuple[torch.Tensor, ...], dict[str, torch.Tensor]], None
]


def _cos_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    """1 - cosine similarity in float64, clamped at 0.

    Near-identical inputs can produce 1 - cos slightly below zero from
    accumulation rounding, which would break the multiplicative oracle
    margin; the clamp keeps distances well-ordered.
    """
    af = a.to(torch.float64).flatten()
    bf = b.to(torch.float64).flatten()
    cos = torch.dot(af, bf) / (torch.linalg.norm(af) * torch.linalg.norm(bf))
    return max(float(1.0 - cos), 0.0)


def _tokenize_prompts() -> list[np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(TARGET_REPO_ID)
    return [np.asarray(tokenizer(p).input_ids, dtype=np.int64) for p in PROMPTS]


def _torch_layer_input_dumps(
    prompt_ids: list[np.ndarray], dtype: torch.dtype, device: str
) -> dict[int, torch.Tensor]:
    """Dumps the tensors entering decoder layers ``TORCH_TAP_LAYERS``.

    Forward pre-hooks capture exactly what vLLM's aux capture denotes: the
    raw residual stream at each layer's input, before its input_layernorm.
    Per-prompt dumps are concatenated in prompt order, matching the ragged
    layout of the MAX pass. The model is freed before returning.
    """
    model = Gemma4ForConditionalGeneration.from_pretrained(
        TARGET_REPO_ID,
        dtype=dtype,
        device_map=device,
    )
    model.eval()
    language_model = model.model.language_model

    captured: dict[int, list[torch.Tensor]] = {j: [] for j in TORCH_TAP_LAYERS}
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(layer_id: int) -> _PreHook:
        def hook(
            module: object,
            args: tuple[torch.Tensor, ...],
            kwargs: dict[str, torch.Tensor],
        ) -> None:
            hidden = args[0] if args else kwargs["hidden_states"]
            captured[layer_id].append(hidden.detach()[0].to("cpu").float())

        return hook

    for j in TORCH_TAP_LAYERS:
        hooks.append(
            language_model.layers[j].register_forward_pre_hook(
                make_hook(j), with_kwargs=True
            )
        )

    with torch.no_grad():
        for ids in prompt_ids:
            language_model(
                input_ids=torch.from_numpy(ids).unsqueeze(0).to(device),
                use_cache=False,
            )

    for h in hooks:
        h.remove()
    dumps = {j: torch.cat(captured[j], dim=0) for j in TORCH_TAP_LAYERS}

    del language_model, model
    gc.collect()
    torch.cuda.empty_cache()
    return dumps


def _build_max_text_config(
    config_json: dict[str, Any], devices: list[DeviceRef]
) -> Gemma4ForConditionalGenerationConfig:
    tc = config_json["text_config"]
    layer_types: list[str] = tc["layer_types"]
    sliding_layers = sum(1 for t in layer_types if t == "sliding_attention")
    global_layers = sum(1 for t in layer_types if t == "full_attention")
    global_rope = tc["rope_parameters"]["full_attention"]
    sliding_rope = tc["rope_parameters"]["sliding_attention"]
    assert global_rope["rope_type"] == "proportional"
    assert sliding_rope["rope_type"] == "default"
    activation_map = {"gelu_pytorch_tanh": "gelu_tanh"}

    sliding_kv = MHAKVCacheParams(
        dtype=MAX_DTYPE,
        n_kv_heads=tc["num_key_value_heads"],
        head_dim=tc["head_dim"],
        num_layers=sliding_layers,
        devices=devices,
        page_size=128,
    )
    global_kv = MHAKVCacheParams(
        dtype=MAX_DTYPE,
        n_kv_heads=tc["num_global_key_value_heads"],
        head_dim=tc["global_head_dim"],
        num_layers=global_layers,
        devices=devices,
        page_size=128,
    )
    kv_params = MultiKVCacheParams.from_params(
        {"sliding_attention": sliding_kv, "full_attention": global_kv}
    )
    text_kv = MHAKVCacheParams(
        dtype=MAX_DTYPE,
        n_kv_heads=tc["num_key_value_heads"],
        head_dim=tc["head_dim"],
        num_layers=tc["num_hidden_layers"],
        devices=devices,
        page_size=128,
    )
    text_config = Gemma4TextConfig(
        vocab_size=tc["vocab_size"],
        hidden_size=tc["hidden_size"],
        intermediate_size=tc["intermediate_size"],
        num_hidden_layers=tc["num_hidden_layers"],
        num_attention_heads=tc["num_attention_heads"],
        num_key_value_heads=tc["num_key_value_heads"],
        head_dim=tc["head_dim"],
        hidden_activation=activation_map.get(
            tc["hidden_activation"], tc["hidden_activation"]
        ),
        max_position_embeddings=tc["max_position_embeddings"],
        max_seq_len=4096,
        rms_norm_eps=tc["rms_norm_eps"],
        rope_theta=-1,
        rope_scaling=None,
        attention_bias=tc["attention_bias"],
        sliding_window=tc["sliding_window"],
        final_logit_softcapping=tc["final_logit_softcapping"],
        attn_logit_softcapping=None,
        rope_local_base_freq=sliding_rope["rope_theta"],
        sliding_window_pattern=-1,
        dtype=MAX_DTYPE,
        devices=devices,
        interleaved_rope_weights=False,
        kv_params=text_kv,
        num_global_key_value_heads=tc["num_global_key_value_heads"],
        global_head_dim=tc["global_head_dim"],
        attention_k_eq_v=tc["attention_k_eq_v"],
        global_rope_scaling=ProportionalScalingParams(
            partial_rotary_factor=global_rope["partial_rotary_factor"]
        ),
        global_rope_theta=global_rope["rope_theta"],
        sliding_window_rope_theta=sliding_rope["rope_theta"],
        layer_types=layer_types,
        return_hidden_states=ReturnHiddenStates.SELECTED_LAYERS,
        target_layer_ids=list(MAX_CAPTURE_IDS),
    )
    return Gemma4ForConditionalGenerationConfig(
        devices=devices,
        dtype=MAX_DTYPE,
        kv_params=kv_params,
        image_token_index=config_json["image_token_id"],
        text_config=text_config,
        vision_config=None,
        tie_word_embeddings=config_json["tie_word_embeddings"],
    )


def _max_layer_output_captures(
    prompt_ids: list[np.ndarray],
) -> dict[int, torch.Tensor]:
    """Runs the MAX 31B target with SELECTED_LAYERS at ``MAX_CAPTURE_IDS``.

    Mirrors ``gemma4/model.py:_build_language_graph`` with the vision
    inputs replaced by in-graph zero-row constants (the text-only pattern of
    ``unified_dspark_gemma4_31b.py:_empty_vision_inputs``). Returns the captured
    layer-output residual streams keyed by MAX layer id, as fp32 CPU
    tensors of shape ``[total_seq_len, hidden]`` in prompt order.
    """
    device = Accelerator()
    session = InferenceSession(devices=[device])
    device_ref = DeviceRef.GPU()

    snapshot_dir = Path(
        snapshot_download(
            TARGET_REPO_ID,
            allow_patterns=["*.safetensors", "*.json"],
        )
    )
    with open(snapshot_dir / "config.json") as f:
        config_json = json.load(f)
    config = _build_max_text_config(config_json, [device_ref])
    hidden_size = config.text_config.hidden_size

    weights = load_weights(sorted(snapshot_dir.glob("*.safetensors")))
    lang_state_dict = convert_safetensor_language_state_dict(
        dict(weights.items())
    )

    kv_params = config.kv_params
    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=16,
        session=session,
        max_batch_size=len(prompt_ids),
    )

    tokens_type = TensorType(
        DType.int64, shape=["total_seq_len"], device=device_ref
    )
    offsets_type = TensorType(
        DType.uint32, shape=["input_row_offsets_len"], device=device_ref
    )
    return_n_logits_type = TensorType(
        DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
    )
    signals = Signals(devices=[device_ref])

    with Graph(
        "gemma4_31b_tap_parity",
        input_types=(
            tokens_type,
            offsets_type,
            return_n_logits_type,
            *signals.input_types(),
            *kv_params.flattened_kv_inputs(),
        ),
    ) as graph:
        model = Gemma4TextModel(config)
        model.load_state_dict(lang_state_dict, weight_alignment=1, strict=True)

        it = iter(graph.inputs)
        tokens = next(it).tensor
        offsets = next(it).tensor
        return_n_logits = next(it).tensor
        signal_buffers = [next(it).buffer]
        sliding_kv, global_kv = kv_params.unflatten_basic_kv_tree(it)

        empty_embeds = ops.constant(
            np.zeros((0, hidden_size), dtype=np.float32),
            DType.float32,
            device=device_ref,
        ).cast(config.unquantized_dtype)
        empty_indices = ops.range(
            0, 0, 1, out_dim=0, dtype=DType.int32, device=device_ref
        )

        outputs = model(
            tokens,
            signal_buffers,
            sliding_kv,
            global_kv,
            return_n_logits,
            [offsets],
            [empty_embeds],
            [empty_indices],
        )
        graph.output(*outputs)

    compiled = session.load(graph, weights_registry=model.state_dict())

    lens = [len(ids) for ids in prompt_ids]
    offsets_np = np.cumsum([0] + lens).astype(np.uint32)
    tokens_np = np.concatenate(prompt_ids).astype(np.int64)
    contexts = [create_text_context(ids, max_length=4096) for ids in prompt_ids]
    for context in contexts:
        kv_manager.claim(context)
    try:
        for context in contexts:
            kv_manager.alloc(context)
        kv_inputs = kv_manager.runtime_inputs(
            [contexts], max_cache_length=max(lens)
        )
        outs = compiled.execute(
            Buffer.from_numpy(tokens_np).to(device),
            Buffer.from_numpy(offsets_np).to(device),
            Buffer.from_numpy(np.array([1], dtype=np.int64)),
            Buffer.zeros(
                shape=(Signals.NUM_BYTES,), dtype=DType.uint8, device=device
            ),
            *kv_inputs.flatten(),
        )
    finally:
        for context in contexts:
            kv_manager.release(context)

    # outputs = (last_logits, *captures) with LAST_TOKEN logits; captures
    # come back one per layer id, ascending (single device).
    assert len(outs) == 1 + len(MAX_CAPTURE_IDS)
    return {
        layer_id: from_dlpack(outs[1 + i]).to("cpu").float()
        for i, layer_id in enumerate(MAX_CAPTURE_IDS)
    }


def test_target_tap_parity() -> None:
    if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
        pytest.skip("HF Hub offline mode is enabled")
    prompt_ids = _tokenize_prompts()

    # Sequential passes: torch bf16 on GPU (~62 GiB, freed before MAX
    # loads), the fp32 oracle on host RAM (~126 GiB, GPU-free), then MAX.
    ref_bf16 = _torch_layer_input_dumps(prompt_ids, torch.bfloat16, "cuda")
    oracle_fp32 = _torch_layer_input_dumps(prompt_ids, torch.float32, "cpu")
    max_captures = _max_layer_output_captures(prompt_ids)

    total_len = sum(len(ids) for ids in prompt_ids)
    ref_shape = ref_bf16[TORCH_TAP_LAYERS[0]].shape
    assert ref_shape[0] == total_len
    for j in TORCH_TAP_LAYERS:
        assert ref_bf16[j].shape == ref_shape
    for k in MAX_CAPTURE_IDS:
        assert max_captures[k].shape == ref_shape

    positive_pairs = [(j - 1, j) for j in VLLM_AUX_LAYER_IDS]
    adjacency_pair = (1, ADJACENCY_TORCH_LAYER)
    positive_cos: dict[int, float] = {}
    ref_oracle_cos: dict[int, float] = {}

    for max_id, torch_id in positive_pairs + [adjacency_pair]:
        cos_bf16 = _cos_dist(max_captures[max_id], ref_bf16[torch_id])
        cos_max_oracle = _cos_dist(max_captures[max_id], oracle_fp32[torch_id])
        cos_ref_oracle = _cos_dist(ref_bf16[torch_id], oracle_fp32[torch_id])
        positive_cos[torch_id] = cos_bf16
        ref_oracle_cos[torch_id] = cos_ref_oracle
        print(
            f"[tap max={max_id} vs torch-input={torch_id}]"
            f" cos_dist bf16={cos_bf16:.3e}"
            f" | vs fp32 oracle: max={cos_max_oracle:.3e}"
            f" ref_bf16={cos_ref_oracle:.3e}"
        )
        assert cos_max_oracle < ORACLE_MARGIN * cos_ref_oracle + 1e-4, (
            f"MAX layer-output {max_id} vs torch layer-input {torch_id}:"
            f" MAX-vs-fp32 {cos_max_oracle:.3e} is worse than"
            f" {ORACLE_MARGIN}x the bf16 reference's own error"
            f" {cos_ref_oracle:.3e}"
        )
        assert cos_bf16 < HS_COS_DIST_BACKSTOP, (
            f"MAX layer-output {max_id} vs torch layer-input {torch_id}:"
            f" cos_dist {cos_bf16:.3e} exceeds {HS_COS_DIST_BACKSTOP}"
        )

    # Negative control (assert-inverted): wiring the checkpoint's aux ids
    # id-for-id into MAX — capture j instead of j-1 — must fail BOTH
    # positive gates (the inverted oracle margin and the absolute
    # backstop) and sit far from the matched pair, or this test could not
    # catch the off-by-one it exists to prevent.
    for j in VLLM_AUX_LAYER_IDS:
        cos_neg = _cos_dist(max_captures[j], ref_bf16[j])
        cos_neg_oracle = _cos_dist(max_captures[j], oracle_fp32[j])
        cos_pos = positive_cos[j]
        oracle_gate = ORACLE_MARGIN * ref_oracle_cos[j] + 1e-4
        print(
            f"[negative control max={j} vs torch-input={j}]"
            f" cos_dist={cos_neg:.3e} vs_oracle={cos_neg_oracle:.3e}"
            f" (matched pair: {cos_pos:.3e}, oracle gate: {oracle_gate:.3e})"
        )
        assert cos_neg_oracle > oracle_gate, (
            f"off-by-one wiring at layer {j} would pass the oracle-margin"
            f" gate: {cos_neg_oracle:.3e} <= {oracle_gate:.3e}"
        )
        assert cos_neg > HS_COS_DIST_BACKSTOP, (
            f"off-by-one wiring at layer {j} would pass the absolute"
            f" backstop: {cos_neg:.3e}"
        )
        assert cos_neg > NEGATIVE_SEPARATION * max(cos_pos, 1e-6), (
            f"off-by-one wiring at layer {j} is not separable from the"
            f" correct wiring: {cos_neg:.3e} vs {cos_pos:.3e}"
        )
