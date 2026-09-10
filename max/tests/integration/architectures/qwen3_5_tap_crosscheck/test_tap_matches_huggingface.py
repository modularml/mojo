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
"""Qwen3.5's DFlash2 tap, checked against HuggingFace rather than itself.

`unified_dflash2_qwen3_5/test_tap_capture.py` pins the tap against MAX's own
final hidden state. That rules out a hook firing at the wrong point in the
loop, but it cannot see a disagreement between MAX's layer `i` and
HuggingFace's layer `i` -- and the DFlash2 reference reads
`hidden_states[layer_id + 1]` from HuggingFace, so that correspondence is
what the drafter's context actually rests on.

An off-by-one here raises nothing. The drafter still runs, still emits
tokens, and the only symptom is a quietly lower acceptance length.

The discriminating test is therefore not equality against one index but a
*ranking* across three: a tap at layer `N` is compared against HuggingFace's
`N`, `N + 1` and `N + 2`, and `N + 1` has to be the clear winner. Both
models run float32 with identical weights, so the intended index matches to
kernel noise and its neighbours miss by whole layers.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph
from max.nn.comm.allreduce import Signals
from max.nn.kv_cache import KVCacheInputs, MHAKVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.qwen3_5.model_config import Qwen3_5Config
from max.pipelines.architectures.qwen3_5.qwen3_5 import Qwen3_5
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.kv_cache import PagedKVCacheManager
from max.pipelines.modeling.types import RequestID
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

SEED = 20260822
HIDDEN = 64
VOCAB = 128
NUM_LAYERS = 8
NUM_HEADS = 2
NUM_KV_HEADS = 1
HEAD_DIM = 32
INTERMEDIATE = 128
# The gated-delta recurrence kernel is only compiled for 128 x 128 heads.
LINEAR_KEY_DIM = 128
LINEAR_VALUE_DIM = 128
LINEAR_KEY_HEADS = 1
LINEAR_VALUE_HEADS = 2
CONV_KERNEL = 4
PARTIAL_ROTARY = 0.25
ROPE_THETA = 1e7
RMS_EPS = 1e-6
SEQ_LEN = 6
PAGE_SIZE = 32

# `full_attention_interval` 4 puts full attention on 3 and 7 -- the same
# residue as every full-attention layer of the real 64-layer model, where the
# taps 5/19/33/47/61 straddle both kinds.
LAYER_TYPES = [
    "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
    for i in range(NUM_LAYERS)
]
NUM_FULL = sum(t == "full_attention" for t in LAYER_TYPES)
NUM_LINEAR = sum(t == "linear_attention" for t in LAYER_TYPES)
CONV_DIM = 2 * LINEAR_KEY_HEADS * LINEAR_KEY_DIM + (
    LINEAR_VALUE_HEADS * LINEAR_VALUE_DIM
)
TOKENS = np.arange(SEQ_LEN, dtype=np.int64)


def _hf_config() -> Qwen3_5TextConfig:
    section = HEAD_DIM // 8
    config = Qwen3_5TextConfig(
        hidden_size=HIDDEN,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        num_hidden_layers=NUM_LAYERS,
        intermediate_size=INTERMEDIATE,
        vocab_size=VOCAB,
        full_attention_interval=4,
        linear_key_head_dim=LINEAR_KEY_DIM,
        linear_value_head_dim=LINEAR_VALUE_DIM,
        linear_num_key_heads=LINEAR_KEY_HEADS,
        linear_num_value_heads=LINEAR_VALUE_HEADS,
        linear_conv_kernel_dim=CONV_KERNEL,
        partial_rotary_factor=PARTIAL_ROTARY,
        rms_norm_eps=RMS_EPS,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": ROPE_THETA,
            "mrope_section": [section, section, HEAD_DIM // 4 - 2 * section],
            "mrope_interleaved": True,
            "partial_rotary_factor": PARTIAL_ROTARY,
        },
    )
    assert list(config.layer_types) == LAYER_TYPES, config.layer_types
    return config


@torch.inference_mode()
def _huggingface() -> tuple[
    dict[str, np.ndarray], list[np.ndarray], np.ndarray
]:
    """Returns the weights, every hidden-state entry, and the normed tail."""
    torch.manual_seed(SEED)
    model = Qwen3_5TextModel(_hf_config()).to(torch.float32).eval()
    # Default init leaves norm weights at exactly 1.0, which makes different
    # layers' outputs coincide more than they should and would soften the
    # very off-by-one this test exists to catch. Randomise everything.
    generator = torch.Generator().manual_seed(SEED)
    for name, param in model.named_parameters():
        if name.endswith(("A_log", "dt_bias")):
            # These feed exp()/softplus(); keep them in a sane range.
            param.copy_(
                torch.rand(param.shape, generator=generator) * 0.5 + 0.25
            )
        else:
            param.copy_(torch.randn(param.shape, generator=generator) * 0.05)

    ids = torch.tensor(TOKENS).unsqueeze(0)
    out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
    weights = {k: v.numpy().copy() for k, v in model.state_dict().items()}
    hidden = [h[0].float().numpy().copy() for h in out.hidden_states]

    # Pin HuggingFace's indexing while the model is in hand: entry 0 is the
    # embedding output, so entry i+1 is the output of layer i.
    embed = model.embed_tokens(ids)[0].float().numpy()
    np.testing.assert_array_equal(hidden[0], embed)
    assert len(hidden) == NUM_LAYERS + 1
    normed = model.norm(out.hidden_states[-1])[0].float().numpy().copy()
    return weights, hidden, normed


def _max_config(
    target_layer_ids: list[int], use_subgraphs: bool = False
) -> Qwen3_5Config:
    device = DeviceRef.GPU()
    config = Qwen3_5Config(
        hidden_size=HIDDEN,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        num_hidden_layers=NUM_LAYERS,
        rope_theta=ROPE_THETA,
        rope_scaling_params=None,
        max_seq_len=128,
        intermediate_size=INTERMEDIATE,
        interleaved_rope_weights=True,
        vocab_size=VOCAB,
        dtype=DType.float32,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=MHAKVCacheParams(
            dtype=DType.float32,
            devices=[device],
            n_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            num_layers=NUM_FULL,
            page_size=PAGE_SIZE,
        ),
        norm_dtype=DType.float32,
        rms_norm_eps=RMS_EPS,
        attention_multiplier=HEAD_DIM**-0.5,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=[device],
        clip_qkv=None,
        layer_types=list(LAYER_TYPES),
        linear_key_head_dim=LINEAR_KEY_DIM,
        linear_value_head_dim=LINEAR_VALUE_DIM,
        linear_num_key_heads=LINEAR_KEY_HEADS,
        linear_num_value_heads=LINEAR_VALUE_HEADS,
        linear_conv_kernel_dim=CONV_KERNEL,
        partial_rotary_factor=PARTIAL_ROTARY,
        use_subgraphs=use_subgraphs,
        state_pool_dtype=DType.float32,
    )
    config.target_layer_ids = target_layer_ids
    return config


def _max_taps(
    weights: dict[str, np.ndarray],
    target_layer_ids: list[int],
    use_subgraphs: bool = False,
) -> list[np.ndarray]:
    """Runs MAX's Qwen3.5 on the HuggingFace weights, returning its captures."""
    config = _max_config(target_layer_ids, use_subgraphs)
    model = Qwen3_5(config)
    model.return_logits = ReturnLogits.LAST_TOKEN
    model.return_hidden_states = ReturnHiddenStates.SELECTED_LAYERS

    rng = np.random.default_rng(0)
    state: dict[str, np.ndarray] = {}
    missing = []
    for name, weight in model.raw_state_dict().items():
        shape = [int(d) for d in weight.shape]
        if name in weights:
            assert list(weights[name].shape) == shape, (
                name,
                weights[name].shape,
                shape,
            )
            state[name] = weights[name].astype(np.float32)
        else:
            # `lm_head` is the only tensor the HF *text* model does not carry.
            # It sits downstream of every tap and cannot affect one.
            missing.append(name)
            state[name] = rng.standard_normal(shape, dtype=np.float32) * 0.05
    assert missing == ["lm_head.weight"], missing

    model.load_state_dict(state, weight_alignment=1, strict=False)
    registry = model.state_dict()

    device = Accelerator()
    session = InferenceSession(devices=[device])
    with Graph(
        "qwen3_5_tap_crosscheck",
        input_types=model.input_types(config.kv_params),
    ) as graph:
        tokens, row_offsets, return_n_logits, *rest = graph.inputs
        it = iter(rest)
        signal_buffers = [next(it).buffer]
        leaf = config.kv_params.unflatten_kv_inputs(it)
        assert isinstance(leaf, KVCacheInputs)
        slot_idx = [next(it).tensor]
        pools = [[next(it).buffer for _ in range(NUM_LINEAR)] for _ in range(2)]
        outputs = model(
            tokens.tensor,
            list(leaf.inputs),
            return_n_logits.tensor,
            row_offsets.tensor,
            signal_buffers,
            slot_idx,
            [pools[0]],
            [pools[1]],
        )
        graph.output(*outputs)

    compiled = session.load(graph, weights_registry=registry)

    def buf(x: np.ndarray) -> Buffer:
        return Buffer.from_numpy(np.ascontiguousarray(x)).to(device)

    kv_manager = PagedKVCacheManager(
        params=config.kv_params,
        total_num_pages=8,
        session=session,
        max_batch_size=2,
    )
    ctx = TextContext(
        request_id=RequestID(), max_length=128, tokens=TokenBuffer(TOKENS)
    )
    kv_manager.claim(ctx)
    kv_manager.alloc(ctx)
    kv_inputs = list(
        kv_manager.runtime_inputs_for_leaf([[ctx]]).inputs[0].flatten()
    )

    results = compiled.execute(
        buf(TOKENS),
        buf(np.array([0, SEQ_LEN], dtype=np.uint32)),
        Buffer.from_numpy(np.array([1], dtype=np.int64)),
        *Signals.allocate([device]),
        *kv_inputs,
        buf(np.array([0], dtype=np.uint32)),
        *[
            buf(np.zeros((2, CONV_DIM, CONV_KERNEL - 1), dtype=np.float32))
            for _ in range(NUM_LINEAR)
        ],
        *[
            buf(
                np.zeros(
                    (2, LINEAR_VALUE_HEADS, LINEAR_KEY_DIM, LINEAR_VALUE_DIM),
                    dtype=np.float32,
                )
            )
            for _ in range(NUM_LINEAR)
        ],
    )
    every = [np.array(r.to(CPU()).to_numpy()) for r in results]
    print(
        "\nraw graph outputs: "
        + ", ".join(
            f"{i}:{a.shape}/{np.abs(a).max():.3e}" for i, a in enumerate(every)
        )
    )
    # LAST_TOKEN logits first, then one capture per tapped layer.
    return every[1:]


def _relative(a: np.ndarray, b: np.ndarray) -> float:
    scale = max(float(np.abs(a).max()), 1e-6)
    return float(np.abs(a - b).max() / scale)


@pytest.fixture(scope="module")
def crosscheck() -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    weights, hidden, normed = _huggingface()
    taps = _max_taps(weights, list(range(NUM_LAYERS)))
    assert len(taps) == NUM_LAYERS
    return taps, hidden, normed


# HuggingFace applies the model's final norm to the last `hidden_states`
# entry and to no other, so `hidden_states[NUM_LAYERS]` is not comparable to
# a raw tap. `test_huggingface_normalises_only_its_last_entry` measures that
# rather than assuming it. The real model taps 5/19/33/47/61 of 64 layers and
# never the last, so this excludes nothing the drafter uses.
COMPARABLE = range(NUM_LAYERS - 1)


def test_huggingface_normalises_only_its_last_entry(
    crosscheck: tuple[list[np.ndarray], list[np.ndarray], np.ndarray],
) -> None:
    """Why the last layer is excluded from the comparison below.

    Every entry but the last is the raw residual stream. The last one has the
    final norm already applied, which is a HuggingFace convention and not
    something a tap could or should reproduce.
    """
    _, hidden, normed = crosscheck
    raw = [float(np.abs(h).max()) for h in hidden[:-1]]
    last = float(np.abs(hidden[-1]).max())
    print(
        "\nHF hidden_states magnitudes: " + ", ".join(f"{v:.3f}" for v in raw)
    )
    print(f"HF hidden_states[-1]: {last:.3f}")
    # The raw entries grow smoothly; the normed one jumps clear of them.
    assert last > 4 * max(raw)
    # And re-normalising it barely moves it, because it is already normed.
    assert _relative(hidden[-1], normed) < 0.2


def test_every_tap_matches_its_own_layer_and_not_its_neighbours(
    crosscheck: tuple[list[np.ndarray], list[np.ndarray], np.ndarray],
) -> None:
    """The ranking, not the equality, is the finding.

    A tap at layer ``N`` is meant to be HuggingFace's ``hidden_states[N + 1]``.
    Its neighbours are whole layers away, so the intended index has to win by
    orders of magnitude. Asserting only "matches ``N + 1`` closely" would pass
    for a model whose consecutive layers happen to be similar; requiring the
    neighbours to lose by 50x cannot.
    """
    taps, hidden, _ = crosscheck
    rows = []
    for layer in COMPARABLE:
        scores = {
            offset: _relative(hidden[layer + offset], taps[layer])
            for offset in (0, 1, 2)
        }
        best = min(scores, key=lambda k: scores[k])
        runner_up = min(v for k, v in scores.items() if k != 1)
        rows.append((layer, scores, runner_up / scores[1]))
        assert best == 1, (
            f"tap at layer {layer} is closer to hidden_states[{layer + best}] "
            f"than to the intended hidden_states[{layer + 1}]: {scores}"
        )
        assert runner_up / scores[1] > 50, (
            f"layer {layer}: intended index wins by only "
            f"{runner_up / scores[1]:.1f}x, too little to call: {scores}"
        )
    print("\nlayer   rel@N     rel@N+1   rel@N+2   separation")
    for layer, scores, ratio in rows:
        print(
            f"{layer:>5}   {scores[0]:.2e}  {scores[1]:.2e}  "
            f"{scores[2]:.2e}  {ratio:>7.0f}x"
        )


def test_taps_are_not_permuted(
    crosscheck: tuple[list[np.ndarray], list[np.ndarray], np.ndarray],
) -> None:
    """Capture `k` holds layer `k`'s output, across the whole stack.

    `fc` concatenates one hidden-wide block per tap in ascending layer order,
    so a permuted capture list feeds every block the wrong layer's states and
    surfaces only as lost acceptance.
    """
    taps, hidden, _ = crosscheck
    layers = list(COMPARABLE)
    matrix = np.array(
        [[_relative(hidden[j + 1], taps[i]) for j in layers] for i in layers]
    )
    print("\nrelative error, MAX tap (row) vs HF layer output (column)")
    print("      " + "".join(f"{j:>10}" for j in layers))
    for i, row in zip(layers, matrix, strict=True):
        print(f"tap {i} " + "".join(f"{v:>10.2e}" for v in row))
    np.testing.assert_array_equal(matrix.argmin(axis=1), np.arange(len(layers)))


def test_the_real_tap_pattern_reads_the_same_layers(
    crosscheck: tuple[list[np.ndarray], list[np.ndarray], np.ndarray],
) -> None:
    """A scattered request, like the real config's, is not reordered.

    Capture order comes from the layer loop rather than from
    `target_layer_ids`, so asking for a handful of scattered layers has to
    return them in ascending order and read the right ones.
    """
    _, hidden, _ = crosscheck
    weights, _, _ = _huggingface()
    scattered = [1, 3, 5, 6]
    taps = _max_taps(weights, scattered)
    assert len(taps) == len(scattered)
    for layer, tap in zip(scattered, taps, strict=True):
        best = min(
            range(NUM_LAYERS - 1),
            key=lambda j: _relative(hidden[j + 1], tap),
        )
        assert best == layer, f"scattered tap for {layer} read layer {best}"


def test_the_subgraph_path_taps_the_same_layers() -> None:
    """`forward_sequential_layers` calls the hook from two places.

    The DFlash2 target config sets `use_subgraphs = False`, so production
    takes the plain branch, but the subgraph branch has its own
    `on_layer_output` call and would silently diverge if it ever got turned
    on. Same weights, same expectation.
    """
    weights, hidden, _ = _huggingface()
    taps = _max_taps(weights, list(COMPARABLE), use_subgraphs=True)
    for layer in COMPARABLE:
        best = min(
            COMPARABLE, key=lambda j: _relative(hidden[j + 1], taps[layer])
        )
        assert best == layer, (
            f"with subgraphs, tap for layer {layer} read layer {best}"
        )
