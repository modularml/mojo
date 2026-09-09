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
"""Tests for max.nn.sampling"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import CPU, Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import (
    topk_fused_sampling_with_dist,
    topk_topp_masked_probs,
)
from max.nn.sampling import (
    MinPSampler,
    compute_synthetic_acceptance_base_rate,
    greedy_acceptance_sampler,
    stochastic_acceptance_sampler,
)
from max.pipelines.sampling import (
    SyntheticRunner,
    build_greedy_acceptance_sampler_graph,
    build_stochastic_acceptance_sampler_graph,
    build_synthetic_acceptance_sampler_graph,
)
from max.support.math import ceildiv


def _seed_buffer(value: int, device: Device | None = None) -> Buffer:
    buf = Buffer.from_numpy(np.array([value], dtype=np.uint64))
    if device is not None:
        buf = buf.to(device)
    return buf


def _pack_bitmask(mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.int32]:
    """Pack a bool bitmask ``[..., vocab]`` into int32 words ``[..., ceil(vocab/32)]``.

    Bit ``v`` lives in word ``v // 32`` at bit position ``v % 32`` -- the packed
    layout ``apply_packed_bitmask`` consumes. Padding bits past ``vocab`` stay
    zero; the kernel never reads them.
    """
    mask = np.asarray(mask, dtype=bool)
    *lead, vocab = mask.shape
    packed_vocab = ceildiv(vocab, 32)
    pad = packed_vocab * 32 - vocab
    if pad:
        mask = np.concatenate(
            [mask, np.zeros((*lead, pad), dtype=bool)], axis=-1
        )
    bits = mask.reshape(*lead, packed_vocab, 32).astype(np.uint32)
    weights = np.uint32(1) << np.arange(32, dtype=np.uint32)
    words = (bits * weights).sum(axis=-1, dtype=np.uint64).astype(np.uint32)
    return words.view(np.int32)


# NOTE THAT ONLY RANK 2 TENSORS
# ARE CURRENTLY SUPPORTED
@pytest.mark.parametrize(
    ("input_shape", "min_p", "temperature"),
    [
        ((5, 5), 0.0, 0.0),
        ((3, 5), 0.1, 0.3),
    ],
)
def test_min_p_execution(
    session: InferenceSession,
    input_shape: tuple[int, ...],
    min_p: float,
    temperature: float,
) -> None:
    """Tests end-to-end MinPSampling lowering and execution."""
    with Graph(
        "min_p_test",
        input_types=[
            TensorType(DType.float32, shape=input_shape, device=DeviceRef.CPU())
        ],
    ) as graph:
        inputs, *_ = graph.inputs
        sampler = MinPSampler(DType.float32, input_shape, temperature)
        out = sampler(inputs.tensor, min_p)
        graph.output(out)

        # Compile and execute the graph.
        model = session.load(graph)

        # Generate random input data.
        np_input = np.random.randn(*input_shape).astype(np.float32)

        # Execute MAX model.
        min_p_output, *_ = model.execute(np_input)
        cast(Buffer, min_p_output).to_numpy()


def test_min_p_known_inputs_outputs(session: InferenceSession) -> None:
    """Tests MinP sampling with known inputs and expected behavior."""
    batch_size = 5
    vocab_size = 4
    input_shape = (batch_size, vocab_size)
    temperature = 1.0  # Set to 1 for predictable softmax behavior

    with Graph(
        "min_p_known_test",
        input_types=[
            TensorType(
                DType.float32, shape=input_shape, device=DeviceRef.CPU()
            ),
            TensorType(
                DType.float32, shape=(batch_size,), device=DeviceRef.CPU()
            ),
        ],
    ) as graph:
        inputs, min_p_tensor, *_ = graph.inputs
        sampler = MinPSampler(DType.float32, input_shape, temperature)
        out = sampler(inputs.tensor, min_p_tensor.tensor)
        graph.output(out)
        model = session.load(graph)

        np_input = np.array(
            [
                [-1.0, 1.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 3.0],
                [1.0, 1.0, 1.1, 1.0],
                [0.0, 2.0, 4.0, 1.0],
                [0.0, 0.0, 2.0, 1.0],
            ],
            dtype=np.float32,
        )

        min_p_array = np.array([0.1, 0.1, 0.26, 0.1, 0.1], dtype=np.float32)
        min_p_output, *_ = model.execute(np_input, min_p_array)
        result = cast(Buffer, min_p_output).to_numpy()
        assert (
            result[2, 0] == 2
        )  # 1.1 is the only logit that is greater than 0.26 after softmax


def test_greedy_acceptance_sampler(session: InferenceSession) -> None:
    """Tests greedy_acceptance_sampler (accept iff draft == argmax)."""
    device = session.devices[0]
    graph = build_greedy_acceptance_sampler_graph(
        device=DeviceRef.from_device(device)
    )
    model = session.load(graph)

    batch_size = 2
    num_steps = 4
    vocab_size = 6

    # Batch 0: target argmax = [1, 1, 3, 0, bonus=2], draft = [1, 1, 0, 0] → reject at 2
    # Batch 1: target argmax = [2, 4, 4, 5, bonus=1], draft = [2, 4, 4, 5] → all accepted
    target_logits = np.zeros(
        (batch_size * (num_steps + 1), vocab_size), dtype=np.float32
    )
    target_argmax = [[1, 1, 3, 0, 2], [2, 4, 4, 5, 1]]
    for b in range(batch_size):
        for s in range(num_steps + 1):
            target_logits[b * (num_steps + 1) + s, target_argmax[b][s]] = 10.0

    draft_tokens_np = np.array([[1, 1, 0, 0], [2, 4, 4, 5]], dtype=np.int64)

    first_rejected, target_tokens, bonus_tokens = model.execute(
        Buffer.from_numpy(draft_tokens_np).to(device),
        Buffer.from_numpy(target_logits).to(device),
    )
    first_rejected_np = cast(Buffer, first_rejected).to_numpy()
    target_tokens_np = cast(Buffer, target_tokens).to_numpy()
    bonus_tokens_np = cast(Buffer, bonus_tokens).to_numpy()

    assert first_rejected_np.shape == (batch_size,)
    assert target_tokens_np.shape == (batch_size, num_steps)
    assert bonus_tokens_np.shape == (batch_size, 1)
    np.testing.assert_array_equal(first_rejected_np, [2, num_steps])
    np.testing.assert_array_equal(
        target_tokens_np, [[1, 1, 3, 0], [2, 4, 4, 5]]
    )
    np.testing.assert_array_equal(bonus_tokens_np, [[2], [1]])


@pytest.mark.parametrize("device", [Accelerator(), CPU()])
def test_typical_acceptance_sampler(device: Device) -> None:
    """Tests stochastic mode (accept with prob p_target)."""
    session = InferenceSession(devices=[device])
    graph = build_stochastic_acceptance_sampler_graph(
        device=DeviceRef.from_device(device)
    )
    model = session.load(graph)

    def _make_inputs(
        batch_size: int,
        vocab_size: int,
        draft_tokens_np: npt.NDArray[np.int64],
        logits: npt.NDArray[np.float32],
    ) -> list[Buffer]:
        return [
            Buffer.from_numpy(draft_tokens_np).to(device),
            Buffer.from_numpy(logits).to(device),
            Buffer.from_numpy(np.ones(batch_size, dtype=np.float32)).to(device),
            Buffer.from_numpy(
                np.full(batch_size, vocab_size, dtype=np.int64)
            ).to(device),
            Buffer.from_numpy(np.array(vocab_size, dtype=np.int64)),
            Buffer.from_numpy(np.ones(batch_size, dtype=np.float32)).to(device),
            Buffer.from_numpy(np.array(1.0, dtype=np.float32)),
            _seed_buffer(1, device),
        ]

    vocab_size = 5

    # Test both num_steps = 0 and num_steps = 3
    # With num_steps = 0, some tensors will be empty.
    # The kernels should handle this gracefully and not crash.
    for num_steps in [0, 3]:
        # Case 1: draft token has ~1.0 probability → all accepted
        logits_high = np.full(
            (num_steps + 1, vocab_size), -100.0, dtype=np.float32
        )
        logits_high[:, 2] = 100.0
        draft_high = np.full((1, num_steps), 2, dtype=np.int64)

        first_rejected, recovered, bonus = model.execute(
            *_make_inputs(1, vocab_size, draft_high, logits_high)
        )
        assert cast(Buffer, first_rejected).to_numpy()[0] == num_steps
        assert cast(Buffer, recovered).to_numpy().shape == (1, num_steps)
        assert cast(Buffer, bonus).to_numpy().shape == (1, 1)

        # Case 2: draft token has ~0.0 probability → rejected at position 0
        logits_low = np.full(
            (num_steps + 1, vocab_size), -100.0, dtype=np.float32
        )
        logits_low[:, 0] = 100.0
        draft_low = np.full((1, num_steps), 4, dtype=np.int64)

        first_rejected, _, _ = model.execute(
            *_make_inputs(1, vocab_size, draft_low, logits_low)
        )
        assert cast(Buffer, first_rejected).to_numpy()[0] == 0


def test_stochastic_acceptance_graph_requires_vocab_size() -> None:
    with pytest.raises(ValueError, match="vocab_size is required"):
        build_stochastic_acceptance_sampler_graph(
            device=DeviceRef.CPU(), draft_proposal="sampled"
        )


def test_stochastic_acceptance_sampler_draft_dist_required_iff_sampled() -> (
    None
):
    """``draft_probs_full`` is set iff sampled mode.

    The checks fire before any graph op runs, so no active ``Graph`` context or
    real tensors are needed.
    """
    with pytest.raises(
        ValueError, match="draft_probs_full and vocab_size are required"
    ):
        stochastic_acceptance_sampler(
            draft_tokens=cast(Any, None),
            target_logits=cast(Any, None),
            temperature=cast(Any, None),
            top_k=cast(Any, None),
            max_k=cast(Any, None),
            top_p=cast(Any, None),
            min_top_p=cast(Any, None),
            seed=cast(Any, None),
            draft_proposal="sampled",
            vocab_size=4,
        )

    with pytest.raises(ValueError, match="draft_probs_full must be None"):
        stochastic_acceptance_sampler(
            draft_tokens=cast(Any, None),
            target_logits=cast(Any, None),
            temperature=cast(Any, None),
            top_k=cast(Any, None),
            max_k=cast(Any, None),
            top_p=cast(Any, None),
            min_top_p=cast(Any, None),
            seed=cast(Any, None),
            draft_proposal="argmax",
            draft_probs_full=cast(Any, object()),
        )


def test_bfloat16_sampling_matches_exact_float32_upcast() -> None:
    device = Accelerator()
    session = InferenceSession(devices=[device])
    device_ref = DeviceRef.from_device(device)
    rows = 8
    vocab_size = 257

    input_types = [
        TensorType(DType.float32, [rows, vocab_size], device=device_ref),
        TensorType(DType.int64, [rows], device=device_ref),
        TensorType(DType.float32, [rows], device=device_ref),
        TensorType(DType.float32, [rows], device=device_ref),
        TensorType(DType.uint64, [rows], device=device_ref),
    ]
    with Graph("bfloat16_sampling_handoff", input_types=input_types) as graph:
        source, top_k, temperature, top_p, seed = (
            value.tensor for value in graph.inputs
        )
        native_logits = source.cast(DType.bfloat16)
        legacy_logits = native_logits.cast(DType.float32)
        native_masked = topk_topp_masked_probs(
            native_logits,
            top_k=top_k,
            temperature=temperature,
            top_p=top_p,
        )
        legacy_masked = topk_topp_masked_probs(
            legacy_logits,
            top_k=top_k,
            temperature=temperature,
            top_p=top_p,
        )
        native_tokens, native_dist = topk_fused_sampling_with_dist(
            native_logits,
            top_k=top_k,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        legacy_tokens, legacy_dist = topk_fused_sampling_with_dist(
            legacy_logits,
            top_k=top_k,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        graph.output(
            native_masked,
            legacy_masked,
            native_tokens,
            legacy_tokens,
            native_dist,
            legacy_dist,
        )

    model = session.load(graph)
    rng = np.random.default_rng(20260901)
    input_buffers = [
        Buffer.from_numpy(
            rng.normal(0.0, 2.0, size=(rows, vocab_size)).astype(np.float32)
        ).to(device),
        Buffer.from_numpy(np.full(rows, -1, dtype=np.int64)).to(device),
        Buffer.from_numpy(np.ones(rows, dtype=np.float32)).to(device),
        Buffer.from_numpy(np.full(rows, 0.95, dtype=np.float32)).to(device),
        Buffer.from_numpy(np.arange(rows, dtype=np.uint64) + 99).to(device),
    ]

    outputs = model.capture(92, *input_buffers)
    model.replay(92, *input_buffers)
    arrays = [cast(Buffer, output).to_numpy() for output in outputs]
    for native, legacy in zip(arrays[::2], arrays[1::2], strict=True):
        np.testing.assert_array_equal(native, legacy)


@pytest.mark.parametrize(
    ("base_rate", "expected_first_rejected"),
    [(1.0, "num_steps"), (0.0, 0)],
    ids=["rate_1_all_accepted", "rate_0_all_rejected"],
)
def test_synthetic_acceptance_sampler_degenerate(
    session: InferenceSession,
    base_rate: float,
    expected_first_rejected: int | str,
) -> None:
    """At rate=1.0 every draft token is accepted; at rate=0.0 none are.

    Pins boundary behavior of the threshold comparison and
    ``_find_first_rejected`` against off-by-one regressions.
    """
    device = session.devices[0]
    num_steps = 4
    graph = build_synthetic_acceptance_sampler_graph(
        device=DeviceRef.from_device(device),
        base_acceptance_rate=base_rate,
        num_draft_steps=num_steps,
    )
    model = session.load(graph)

    batch_size = 3
    vocab_size = 6
    target_logits = np.zeros(
        (batch_size * (num_steps + 1), vocab_size), dtype=np.float32
    )
    for b in range(batch_size):
        for s in range(num_steps + 1):
            target_logits[b * (num_steps + 1) + s, 0] = 10.0

    draft_tokens_np = np.ones((batch_size, num_steps), dtype=np.int64)

    first_rejected, _, _ = model.execute(
        Buffer.from_numpy(draft_tokens_np).to(device),
        Buffer.from_numpy(target_logits).to(device),
        _seed_buffer(1, device),
    )
    first_rejected_np = cast(Buffer, first_rejected).to_numpy()
    expected = num_steps if expected_first_rejected == "num_steps" else 0
    np.testing.assert_array_equal(first_rejected_np, [expected] * batch_size)


def test_synthetic_acceptance_sampler_zero_draft_tokens(
    session: InferenceSession,
) -> None:
    """With 0 draft tokens (prefill), first_rejected should be 0."""
    device = session.devices[0]
    num_steps = 4
    graph = build_synthetic_acceptance_sampler_graph(
        device=DeviceRef.from_device(device),
        base_acceptance_rate=1.0,
        num_draft_steps=num_steps,
    )
    model = session.load(graph)

    batch_size = 2
    vocab_size = 6
    target_logits = np.zeros((batch_size * 1, vocab_size), dtype=np.float32)
    target_logits[:, 0] = 10.0
    draft_tokens_np = np.ones((batch_size, 0), dtype=np.int64)

    first_rejected, _, _ = model.execute(
        Buffer.from_numpy(draft_tokens_np).to(device),
        Buffer.from_numpy(target_logits).to(device),
        _seed_buffer(1, device),
    )
    first_rejected_np = cast(Buffer, first_rejected).to_numpy()
    np.testing.assert_array_equal(first_rejected_np, [0] * batch_size)


def test_synthetic_acceptance_sampler_mean_rate(
    session: InferenceSession,
) -> None:
    """Calibrated synthetic sampling converges to the target mean rate.

    Calibration solves for ``base_rate`` so the mean joint acceptance
    probability equals ``target_rate``. The observed fraction of accepted
    draft tokens — ``sum(first_rejected_idx) / (batch * K)`` — is exactly
    that mean, so it should converge under Monte Carlo sampling.

    A fresh seed is bound per call so RNG actually varies across runs;
    otherwise the ensemble collapses to a single deterministic draw.
    """
    device = session.devices[0]
    num_steps = 5
    target_rate = 0.5
    base_rate = compute_synthetic_acceptance_base_rate(target_rate, num_steps)
    graph = build_synthetic_acceptance_sampler_graph(
        device=DeviceRef.from_device(device),
        base_acceptance_rate=base_rate,
        num_draft_steps=num_steps,
    )
    model = session.load(graph)

    batch_size = 64
    vocab_size = 6
    num_runs = 50
    total_accepted = 0
    total_tokens = 0

    target_logits = np.zeros(
        (batch_size * (num_steps + 1), vocab_size), dtype=np.float32
    )
    target_logits[:, 0] = 10.0
    draft_tokens_np = np.ones((batch_size, num_steps), dtype=np.int64)

    for run_idx in range(num_runs):
        first_rejected, _, _ = model.execute(
            Buffer.from_numpy(draft_tokens_np).to(device),
            Buffer.from_numpy(target_logits).to(device),
            _seed_buffer(run_idx + 1, device),
        )
        accepted = cast(Buffer, first_rejected).to_numpy()
        total_accepted += accepted.sum()
        total_tokens += batch_size * num_steps

    observed_rate = total_accepted / total_tokens
    assert abs(observed_rate - target_rate) < 0.02


def test_synthetic_runner_mean_rate(
    session: InferenceSession,
) -> None:
    """SyntheticRunner converges to the configured rate over many calls.

    Exercises the full classical-Eagle integration path: calibration,
    graph construction, per-call seed binding, and the
    :class:`RejectionRunner` protocol.
    """
    device = session.devices[0]
    num_steps = 3
    target_rate = 0.5

    runner = SyntheticRunner(
        session=session,
        device_ref=DeviceRef.from_device(device),
        synthetic_acceptance_rate=target_rate,
        num_speculative_tokens=num_steps,
    )

    batch_size = 32
    vocab_size = 6
    num_runs = 40

    target_logits = np.zeros(
        (batch_size * (num_steps + 1), vocab_size), dtype=np.float32
    )
    target_logits[:, 0] = 10.0
    draft_tokens_np = np.ones((batch_size, num_steps), dtype=np.int64)

    total_accepted = 0
    total_tokens = 0
    first_outputs: list[npt.NDArray[np.int64]] = []
    for _ in range(num_runs):
        first_rejected, _, _ = runner.run(
            draft_tokens=Buffer.from_numpy(draft_tokens_np).to(device),
            draft_logits=None,
            target_logits=Buffer.from_numpy(target_logits).to(device),
            target_logit_offsets=Buffer.from_numpy(
                np.zeros(1, dtype=np.uint32)
            ),
            all_draft_logits=None,
            context_batch=[],
        )
        accepted = cast(Buffer, first_rejected).to_numpy().astype(np.int64)
        total_accepted += int(accepted.sum())
        total_tokens += batch_size * num_steps
        if len(first_outputs) < 2:
            first_outputs.append(accepted.copy())

    observed_rate = total_accepted / total_tokens
    assert abs(observed_rate - target_rate) < 0.03

    assert not np.array_equal(first_outputs[0], first_outputs[1])


def _stochastic_sampler_inputs(
    device: Device,
    batch_size: int,
    vocab_size: int,
    draft_tokens_np: npt.NDArray[np.int64],
    logits: npt.NDArray[np.float32],
    temperature_np: npt.NDArray[np.float32],
    top_k_np: npt.NDArray[np.int64] | None = None,
    top_p_np: npt.NDArray[np.float32] | None = None,
    seed: int = 1,
) -> list[Buffer]:
    if top_k_np is None:
        top_k_np = np.full(batch_size, vocab_size, dtype=np.int64)
    if top_p_np is None:
        top_p_np = np.ones(batch_size, dtype=np.float32)
    return [
        Buffer.from_numpy(draft_tokens_np).to(device),
        Buffer.from_numpy(logits).to(device),
        Buffer.from_numpy(temperature_np).to(device),
        Buffer.from_numpy(top_k_np).to(device),
        Buffer.from_numpy(np.array(int(top_k_np.max()), dtype=np.int64)),
        Buffer.from_numpy(top_p_np).to(device),
        Buffer.from_numpy(np.array(float(top_p_np.min()), dtype=np.float32)),
        _seed_buffer(seed, device),
    ]


_TOP_P_GAP_PROBS = np.array([0.85, 0.10, 0.049, 0.001], dtype=np.float64)
_TOP_P_GAP_TOP_P = 0.5
_TOP_P_GAP_DRAFT_TOKEN = 2


def _top_p_gap_inputs(
    device: Device, batch_size: int, seed: int
) -> list[Buffer]:
    """Sampler inputs whose top_p=0.5 nucleus is exactly {token 0}.

    The draft token (2) carries ~5% unfiltered probability, but zero
    probability under any top_p-filtered distribution: a top_p-respecting
    sampler can only ever emit token 0.
    """
    num_steps = 1
    vocab_size = _TOP_P_GAP_PROBS.size
    logits_row = np.log(_TOP_P_GAP_PROBS).astype(np.float32)
    logits = np.tile(logits_row, (batch_size * (num_steps + 1), 1))
    draft_tokens = np.full(
        (batch_size, num_steps), _TOP_P_GAP_DRAFT_TOKEN, dtype=np.int64
    )
    return _stochastic_sampler_inputs(
        device,
        batch_size,
        vocab_size,
        draft_tokens,
        logits,
        np.ones(batch_size, dtype=np.float32),
        top_p_np=np.full(batch_size, _TOP_P_GAP_TOP_P, dtype=np.float32),
        seed=seed,
    )


@pytest.mark.parametrize("device", [Accelerator(), CPU()])
def test_stochastic_acceptance_bonus_token_honors_top_p(
    device: Device,
) -> None:
    """Bonus tokens go through ``topk_fused_sampling``, so tokens outside
    the top_p nucleus are never emitted there."""
    session = InferenceSession(devices=[device])
    model = session.load(
        build_stochastic_acceptance_sampler_graph(
            device=DeviceRef.from_device(device), draft_proposal="argmax"
        )
    )
    batch_size = 64
    for seed in range(1, 17):
        _, _, bonus = model.execute(
            *_top_p_gap_inputs(device, batch_size, seed)
        )
        bonus_np = cast(Buffer, bonus).to_numpy()
        assert np.all(bonus_np == 0), (
            f"bonus token escaped the top_p nucleus: {np.unique(bonus_np)}"
        )


@pytest.mark.parametrize("device", [Accelerator(), CPU()])
def test_stochastic_acceptance_honors_top_p_for_draft_tokens(
    device: Device,
) -> None:
    """A draft token outside the top_p nucleus must never be committed."""
    session = InferenceSession(devices=[device])
    model = session.load(
        build_stochastic_acceptance_sampler_graph(
            device=DeviceRef.from_device(device), draft_proposal="argmax"
        )
    )
    batch_size = 64
    num_steps = 1
    accepted = 0
    for seed in range(1, 17):
        first_rejected, _, _ = model.execute(
            *_top_p_gap_inputs(device, batch_size, seed)
        )
        first_rejected_np = cast(Buffer, first_rejected).to_numpy()
        accepted += int((first_rejected_np == num_steps).sum())
    assert accepted == 0, (
        f"{accepted}/1024 draft positions committed a token outside the "
        "top_p nucleus"
    )


@pytest.mark.parametrize("device", [Accelerator(), CPU()])
def test_stochastic_acceptance_honors_top_p_for_recovered_tokens(
    device: Device,
) -> None:
    """The recovered token committed at a rejection must stay in the nucleus."""
    session = InferenceSession(devices=[device])
    model = session.load(
        build_stochastic_acceptance_sampler_graph(
            device=DeviceRef.from_device(device), draft_proposal="argmax"
        )
    )
    batch_size = 64
    rejected_rows = 0
    outside_nucleus = 0
    for seed in range(1, 17):
        first_rejected, recovered, _ = model.execute(
            *_top_p_gap_inputs(device, batch_size, seed)
        )
        first_rejected_np = cast(Buffer, first_rejected).to_numpy()
        recovered_np = cast(Buffer, recovered).to_numpy()
        committed = recovered_np[first_rejected_np == 0, 0]
        rejected_rows += committed.size
        outside_nucleus += int((committed != 0).sum())
    assert rejected_rows > 0, "expected frequent rejections with p(draft)~=0.05"
    assert outside_nucleus == 0, (
        f"{outside_nucleus}/{rejected_rows} recovered tokens landed outside "
        "the top_p nucleus"
    )


def _sampled_stochastic_sampler_inputs(
    device: Device,
    batch_size: int,
    vocab_size: int,
    draft_tokens_np: npt.NDArray[np.int64],
    logits: npt.NDArray[np.float32],
    temperature_np: npt.NDArray[np.float32],
    draft_probs_np: npt.NDArray[np.float32],
    top_k_np: npt.NDArray[np.int64] | None = None,
    top_p_np: npt.NDArray[np.float32] | None = None,
    seed: int = 1,
    draft_probs_full_np: npt.NDArray[np.float32] | None = None,
) -> list[Buffer]:
    """Builds the sampled-mode graph inputs.

    ``draft_probs_full_np`` defaults to a one-hot distribution carrying
    ``draft_probs_np`` at the drafted token -- the proposal a deterministic
    draft would have made, which reduces the residual to ``p_target`` with the
    drafted token's mass removed. ``q`` is read back out of this tensor, so
    there is no separate scalar input.
    """
    if draft_probs_full_np is None:
        draft_probs_full_np = np.zeros(
            (*draft_tokens_np.shape, vocab_size), dtype=np.float32
        )
        np.put_along_axis(
            draft_probs_full_np,
            draft_tokens_np[..., None],
            draft_probs_np[..., None],
            axis=-1,
        )
    return _stochastic_sampler_inputs(
        device,
        batch_size,
        vocab_size,
        draft_tokens_np,
        logits,
        temperature_np,
        top_k_np=top_k_np,
        top_p_np=top_p_np,
        seed=seed,
    ) + [Buffer.from_numpy(draft_probs_full_np).to(device)]


def test_full_residual_preserves_target_distribution() -> None:
    """With the draft's whole distribution, speculative sampling is exact.

    The textbook guarantee is that a draft proposing from ``q`` and a target
    accepting with ``min(1, p/q)`` -- recovering from ``(p - q)^+`` on
    rejection -- emits tokens distributed exactly as ``p``, for *any* ``q``.
    The scalar-``q`` residual only approximates that, so this is the test
    that separates the two: it uses a deliberately non-deterministic draft
    whose distribution is far from the target's, where the approximation
    visibly skews.
    """
    device = Accelerator()
    session = InferenceSession(devices=[device])

    vocab_size = 6
    num_steps = 1
    batch_size = 8192
    n_executes = 8

    target_logits_row = np.array(
        [2.0, 1.0, 0.5, 0.0, -1.0, 1.5], dtype=np.float32
    )
    target_probs = np.exp(target_logits_row) / np.exp(target_logits_row).sum()
    # A draft distribution deliberately unlike the target's, so a wrong
    # residual shows up as a skew rather than washing out.
    draft_dist = np.array([0.05, 0.05, 0.4, 0.3, 0.15, 0.05], dtype=np.float32)

    per_seq = np.stack([target_logits_row, np.zeros(vocab_size, np.float32)])
    logits = np.tile(per_seq, (batch_size, 1)).astype(np.float32)

    model = session.load(
        build_stochastic_acceptance_sampler_graph(
            device=DeviceRef.from_device(device),
            draft_proposal="sampled",
            vocab_size=vocab_size,
        )
    )

    rng = np.random.default_rng(0)
    committed: list[npt.NDArray[np.int64]] = []
    for seed in range(1, n_executes + 1):
        # Draw each row's draft token from `draft_dist` and pass the exact
        # probability the draft used, mirroring what the draft model emits.
        draft_tokens_np = rng.choice(
            vocab_size, size=(batch_size, num_steps), p=draft_dist
        ).astype(np.int64)
        draft_probs_np = draft_dist[draft_tokens_np].astype(np.float32)
        draft_probs_full_np = np.broadcast_to(
            draft_dist, (batch_size, num_steps, vocab_size)
        ).astype(np.float32)

        _, recovered, _ = model.execute(
            *_sampled_stochastic_sampler_inputs(
                device,
                batch_size,
                vocab_size,
                draft_tokens_np,
                logits,
                np.ones(batch_size, dtype=np.float32),
                draft_probs_np,
                seed=seed,
                draft_probs_full_np=draft_probs_full_np,
            )
        )
        committed.append(cast(Buffer, recovered).to_numpy()[:, 0])

    counts = np.bincount(
        np.concatenate(committed), minlength=vocab_size
    ).astype(np.float64)
    expected = target_probs * counts.sum()
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    # dof = vocab - 1 = 5; the 0.999 critical value is ~20.5.
    assert chi2 < 20.5, (
        f"emitted distribution is not the target's: chi2={chi2:.1f}, "
        f"empirical={counts / counts.sum()}, target={target_probs}"
    )


def test_sampled_acceptance_output_distribution() -> None:
    """Committed tokens must be ~ the top-p-masked target softmax.

    Speculative decoding is lossless only if the committed token at each
    draft position is marginally distributed as the *verification*
    distribution -- here the target softmax masked and renormalized under
    ``top_p=0.8`` -- for any draft proposal with full support. The draft here
    proposes from a noisy full softmax of the target's logits (0-30%
    perturbation), so accept, reject, and residual recovery all fire, and
    the whole GPU stack is in the loop: ``topk_topp_masked_probs`` for the
    target side, the drafted distribution for ``q``, and the gumbel residual
    sampler for recovery. A kernel emitting an unnormalized or unmasked
    distribution skews every bucket by many sigma at this sample size.
    """
    device = Accelerator()
    session = InferenceSession(devices=[device])

    vocab_size = 1024
    num_steps = 4
    batch_size = 512
    n_trials = 32
    top_p = 0.8
    head_count = 5

    rng = np.random.default_rng(0)

    # LLM-shaped target: a sharp head over a lognormal tail (head mass 0.78,
    # so the 0.8 nucleus keeps the whole head plus a sliver of tail).
    head_probs = np.array([0.28, 0.20, 0.13, 0.10, 0.07])
    top_idx = rng.choice(vocab_size, size=head_count, replace=False)
    tail = np.exp(rng.normal(0.0, 1.0, vocab_size))
    tail[top_idx] = 0.0
    tail *= (1.0 - head_probs.sum()) / tail.sum()
    target_probs = tail
    target_probs[top_idx] = head_probs
    target_logits_row = np.log(target_probs).astype(np.float32)

    # The draft proposes from a noisy version of the target's logits --
    # softmaxed with full support, so ``q`` is positive everywhere and the
    # rejection-sampling identity applies for the exactness claim.
    draft_logits_row = target_logits_row * (
        1.0 + rng.uniform(-0.3, 0.3, vocab_size)
    )
    draft_e = np.exp(draft_logits_row - draft_logits_row.max())
    draft_probs_row = draft_e / draft_e.sum()

    # Reference: the top-p-masked renormalized target, by the kernel's own
    # predicate (a token survives iff the mass strictly above it is <= 0.8).
    order = np.argsort(-target_probs)
    mass_above = np.zeros(vocab_size)
    mass_above[order] = np.concatenate(
        ([0.0], np.cumsum(target_probs[order])[:-1])
    )
    survives = mass_above <= top_p
    masked_target = np.where(survives, target_probs, 0.0)
    masked_target /= masked_target.sum()

    logits_np = np.tile(
        target_logits_row, (batch_size * (num_steps + 1), 1)
    ).astype(np.float32)

    model = session.load(
        build_stochastic_acceptance_sampler_graph(
            device=DeviceRef.from_device(device),
            draft_proposal="sampled",
            vocab_size=vocab_size,
        )
    )

    draft_probs_full_np = np.tile(
        draft_probs_row.astype(np.float32), (batch_size, num_steps, 1)
    )
    committed: list[npt.NDArray[np.int64]] = []
    for _ in range(n_trials):
        seed = int(rng.integers(np.iinfo(np.int64).max))
        draft_tokens_np = rng.choice(
            vocab_size, size=(batch_size, num_steps), p=draft_probs_row
        ).astype(np.int64)
        draft_probs_np = draft_probs_row[draft_tokens_np].astype(np.float32)

        _, recovered, _ = model.execute(
            *_sampled_stochastic_sampler_inputs(
                device,
                batch_size,
                vocab_size,
                draft_tokens_np,
                logits_np,
                np.ones(batch_size, dtype=np.float32),
                draft_probs_np,
                top_k_np=np.full(batch_size, -1, dtype=np.int64),
                top_p_np=np.full(batch_size, top_p, dtype=np.float32),
                seed=seed,
                draft_probs_full_np=draft_probs_full_np,
            )
        )
        # In sampled mode ``recovered`` is the committed token per position:
        # the draft token where accepted, the residual sample where rejected.
        committed.append(cast(Buffer, recovered).to_numpy().reshape(-1))

    all_tokens = np.concatenate(committed)
    counts = np.bincount(all_tokens, minlength=vocab_size).astype(np.float64)
    total = counts.sum()

    # Tokens clearly outside the nucleus have zero masked probability and
    # zero residual, so they can never be committed. The margin keeps a
    # float32-vs-float64 boundary flip on the last admitted tail token from
    # tripping an exact-zero assertion.
    definitely_masked = mass_above > top_p + 0.005
    assert counts[definitely_masked].sum() == 0, (
        "committed a token outside the top-p nucleus: "
        f"{np.nonzero(counts * definitely_masked)[0]}"
    )

    # Tail mass: the head holds 0.78/Z of the masked target; an unmasked or
    # unnormalized verification distribution moves this by many sigma at
    # ~65k samples (noise here is ~0.002).
    tail_mass = 1.0 - masked_target[top_idx].sum()
    tail_freq = 1.0 - counts[top_idx].sum() / total
    assert abs(tail_freq - tail_mass) < 0.015, (
        f"committed-token tail mass {tail_freq:.4f} deviates from masked "
        f"target {tail_mass:.4f}"
    )

    # Chi-square over 6 buckets (each head token + aggregated tail).
    # dof = 5; the 0.999 critical value is ~20.5. The threshold adds margin
    # for the per-request shared residual noise rows, which correlate a few
    # same-request positions without biasing any marginal.
    observed = np.append(counts[top_idx], total - counts[top_idx].sum())
    expected = np.append(masked_target[top_idx], tail_mass) * total
    chi2 = float(np.sum((observed - expected) ** 2 / expected))
    assert chi2 < 30.0, (
        f"chi-square {chi2:.1f} over head + tail buckets exceeds 30: "
        f"observed freq {observed / total}, expected {expected / total}"
    )


def test_stochastic_acceptance_sampler_sampled_matches_argmax_when_q_is_one() -> (
    None
):
    """``draft_proposal="sampled"`` with ``draft_probs=1`` must exactly
    match ``draft_proposal="argmax"`` -- the accept test and residual both
    reduce to the same formula when ``q_draft ≡ 1``. This is the load-bearing
    check that every other pipeline's ("argmax") code path is unaffected by
    this change: it exercises the identical machinery with the new
    parameter wired to its identity value.
    """
    device = Accelerator()
    session = InferenceSession(devices=[device])

    vocab_size = 6
    num_steps = 2
    batch_size = 3

    target_argmax = [2, 4, 1]
    draft_tokens_np = np.array([[2, 2], [0, 0], [1, 1]], dtype=np.int64)
    logits = np.full(
        (batch_size * (num_steps + 1), vocab_size), -100.0, dtype=np.float32
    )
    for b, argmax_token in enumerate(target_argmax):
        for s in range(num_steps + 1):
            logits[b * (num_steps + 1) + s, argmax_token] = 100.0

    temperature_np = np.array([0.5, 1.0, 2.5], dtype=np.float32)
    draft_probs_np = np.ones((batch_size, num_steps), dtype=np.float32)

    argmax_graph = build_stochastic_acceptance_sampler_graph(
        device=DeviceRef.from_device(device), draft_proposal="argmax"
    )
    argmax_model = session.load(argmax_graph)
    argmax_out = argmax_model.execute(
        *_stochastic_sampler_inputs(
            device,
            batch_size,
            vocab_size,
            draft_tokens_np,
            logits,
            temperature_np,
        )
    )

    sampled_graph = build_stochastic_acceptance_sampler_graph(
        device=DeviceRef.from_device(device),
        draft_proposal="sampled",
        vocab_size=vocab_size,
    )
    sampled_model = session.load(sampled_graph)
    sampled_out = sampled_model.execute(
        *_sampled_stochastic_sampler_inputs(
            device,
            batch_size,
            vocab_size,
            draft_tokens_np,
            logits,
            temperature_np,
            draft_probs_np,
        )
    )

    for argmax_buf, sampled_buf in zip(argmax_out, sampled_out, strict=True):
        np.testing.assert_array_equal(
            cast(Buffer, argmax_buf).to_numpy(),
            cast(Buffer, sampled_buf).to_numpy(),
        )


def test_sampled_zeroed_distribution_matches_argmax() -> None:
    """A row with no draft distribution must degrade to argmax-mode behavior.

    The overlap scheduler clears a slot it cannot refresh, so a request new to
    the batch arrives with a zeroed row. Reading ``q`` out of that row gives 0,
    which the sampler must treat as "no draft information" rather than as a
    probability -- a literal 0 would make ``coin * q >= p_target`` false and
    accept everything.

    The two modes draw acceptance through different RNG paths (argmax mode
    matches a truncated target sample; sampled mode flips a ratio-test
    coin), so the equivalence is statistical rather than seed-for-seed:
    both accept a draft with its target probability, so their acceptance
    rates must agree. The accept-everything failure this test exists to
    catch would surface as a sampled-mode acceptance rate of ~1.0.
    """
    device = Accelerator()
    session = InferenceSession(devices=[device])
    vocab_size = 8
    batch_size = 512
    num_steps = 2

    rng = np.random.default_rng(21)
    logits = rng.standard_normal(
        (batch_size * (num_steps + 1), vocab_size)
    ).astype(np.float32)
    draft_tokens_np = rng.integers(
        0, vocab_size, size=(batch_size, num_steps)
    ).astype(np.int64)
    temperature_np = np.ones(batch_size, dtype=np.float32)

    sampled = session.load(
        build_stochastic_acceptance_sampler_graph(
            device=DeviceRef.from_device(device),
            draft_proposal="sampled",
            vocab_size=vocab_size,
        )
    )
    argmax = session.load(
        build_stochastic_acceptance_sampler_graph(
            device=DeviceRef.from_device(device), draft_proposal="argmax"
        )
    )

    accepted_at_first_position = {"sampled": 0, "argmax": 0}
    trials = 0
    for seed in range(1, 17):
        first_sampled, recovered_sampled, _ = sampled.execute(
            *_sampled_stochastic_sampler_inputs(
                device,
                batch_size,
                vocab_size,
                draft_tokens_np,
                logits,
                temperature_np,
                np.zeros((batch_size, num_steps), dtype=np.float32),
                seed=seed,
                draft_probs_full_np=np.zeros(
                    (batch_size, num_steps, vocab_size), dtype=np.float32
                ),
            )
        )
        first_argmax, _, _ = argmax.execute(
            *_stochastic_sampler_inputs(
                device,
                batch_size,
                vocab_size,
                draft_tokens_np,
                logits,
                temperature_np,
                seed=seed,
            )
        )
        first_sampled_np = cast(Buffer, first_sampled).to_numpy()
        first_argmax_np = cast(Buffer, first_argmax).to_numpy()
        # P(first_rejected > 0) is the position-0 acceptance rate: one
        # clean Bernoulli per row, unpolluted by later positions.
        accepted_at_first_position["sampled"] += int(
            (first_sampled_np > 0).sum()
        )
        accepted_at_first_position["argmax"] += int((first_argmax_np > 0).sum())
        trials += batch_size

        # Recovery must never re-emit the token the target just rejected --
        # the failure the zero fallback exists to prevent.
        recovered_np = cast(Buffer, recovered_sampled).to_numpy()
        for b in range(batch_size):
            k = int(first_sampled_np[b])
            if k < num_steps:
                assert recovered_np[b, k] != draft_tokens_np[b, k]

    rate_sampled = accepted_at_first_position["sampled"] / trials
    rate_argmax = accepted_at_first_position["argmax"] / trials
    # ~8k trials per arm puts the standard error of the rate difference
    # around 0.005; accept-everything would land rate_sampled at ~1.0.
    assert rate_sampled < 0.9, (
        f"zeroed draft distribution accepted {rate_sampled:.2%} of drafts: "
        "q=0 is being treated as a probability instead of 'no information'"
    )
    assert abs(rate_sampled - rate_argmax) < 0.03, (
        f"zeroed-row sampled mode accepts {rate_sampled:.4f} vs argmax "
        f"{rate_argmax:.4f}: the q=0 fallback no longer degrades to "
        "argmax-mode behavior"
    )


def test_stochastic_acceptance_sampler_sampled_zero_draft_tokens() -> None:
    """A step with no drafts to verify must run, not fault.

    Every prefill-only step hands the verification graph ``num_steps=0``, so
    the sampled path has to survive an empty step axis: the per-step stats
    have no rows at all, and the first-rejected index is 0 with nothing to
    index. Both of these faulted before the residual was reworked to index
    only tensors that are never empty.
    """
    device = Accelerator()
    session = InferenceSession(devices=[device])
    vocab_size = 8
    batch_size = 3
    num_steps = 0

    model = session.load(
        build_stochastic_acceptance_sampler_graph(
            device=DeviceRef.from_device(device),
            draft_proposal="sampled",
            vocab_size=vocab_size,
        )
    )

    rng = np.random.default_rng(3)
    logits = rng.standard_normal(
        (batch_size * (num_steps + 1), vocab_size)
    ).astype(np.float32)

    first_rejected, recovered, bonus = model.execute(
        *_sampled_stochastic_sampler_inputs(
            device,
            batch_size,
            vocab_size,
            np.zeros((batch_size, num_steps), dtype=np.int64),
            logits,
            np.ones(batch_size, dtype=np.float32),
            np.zeros((batch_size, num_steps), dtype=np.float32),
            draft_probs_full_np=np.zeros(
                (batch_size, num_steps, vocab_size), dtype=np.float32
            ),
        )
    )
    assert cast(Buffer, recovered).to_numpy().shape == (batch_size, num_steps)
    # Nothing was drafted, so nothing can be rejected and the bonus token is
    # the only real output.
    np.testing.assert_array_equal(
        cast(Buffer, first_rejected).to_numpy(),
        np.zeros(batch_size, dtype=np.int64),
    )
    bonus_np = cast(Buffer, bonus).to_numpy()
    assert bonus_np.shape == (batch_size, 1)
    assert np.all((bonus_np >= 0) & (bonus_np < vocab_size))


def test_stochastic_acceptance_sampler_sampled_ratio_formula() -> None:
    """Pins the reformulated accept test: ``reject iff coin*q_draft >= p_target``.

    Uses a near-deterministic draft (heavy mass on the drafted token) so the
    residual approximation is exact, isolating the ratio-test numerics.
    """
    device = Accelerator()
    session = InferenceSession(devices=[device])
    graph = build_stochastic_acceptance_sampler_graph(
        device=DeviceRef.from_device(device),
        draft_proposal="sampled",
        vocab_size=4,
    )
    model = session.load(graph)

    vocab_size = 4
    num_steps = 1
    batch_size = 1
    draft_token = 0

    logits = np.full(
        (batch_size * (num_steps + 1), vocab_size), -100.0, dtype=np.float32
    )
    # p_target(draft_token) = 0.5 after softmax over these two logits.
    logits[:, draft_token] = 0.0
    logits[:, draft_token + 1] = 0.0
    draft_tokens_np = np.full((batch_size, num_steps), draft_token, np.int64)

    # q_draft = 0.6 > p_target = 0.5: p_target/q_draft < 1, so this is a
    # real (non-degenerate) ratio test, unlike the argmax q=1 case.
    draft_probs_np = np.full((batch_size, num_steps), 0.6, dtype=np.float32)

    saw_accept = saw_reject = False
    for seed in range(1, 65):
        first_rejected, _, _ = model.execute(
            *_sampled_stochastic_sampler_inputs(
                device,
                batch_size,
                vocab_size,
                draft_tokens_np,
                logits,
                np.ones(batch_size, dtype=np.float32),
                draft_probs_np,
                seed=seed,
            )
        )
        accepted = cast(Buffer, first_rejected).to_numpy()[0] == num_steps
        saw_accept |= bool(accepted)
        saw_reject |= not accepted
    # p_target/q_draft = 0.5/0.6 ≈ 0.833: accept ~83% of draws, so both
    # outcomes must appear across 64 seeds (P(never reject) ≈ 0.83^64 ≈ 6e-6).
    assert saw_accept and saw_reject


def test_stochastic_acceptance_sampler_sampled_degenerate_residual() -> None:
    """A near-all-zero residual must not NaN/crash and must recover a valid
    vocab id, matching vLLM's unguarded Gumbel-max behavior.

    ``draft_token``'s logit dominates so heavily that every other vocab
    entry's softmax mass underflows to ~0 (residual there is already
    ~p_target~0), and ``q_draft`` is deliberately set above 1.0 -- an
    out-of-spec value no real caller should pass, but one the ratio-test
    reformulation must still handle safely (see the function's degrade-
    toward-reject docstring note) -- to force frequent, reliable rejection
    without needing top_k masking (which would make p_target exactly 1 and
    rejection vanishingly rare instead). Exact argmax ties on an all-zero
    row have GPU-nondeterministic tie-break (:func:`ops.argmax`'s
    documented "GPU may return any" caveat), so this only pins finiteness
    and a valid vocab id, not seed-to-seed equality.
    """
    device = Accelerator()
    session = InferenceSession(devices=[device])
    graph = build_stochastic_acceptance_sampler_graph(
        device=DeviceRef.from_device(device),
        draft_proposal="sampled",
        vocab_size=4,
    )
    model = session.load(graph)

    vocab_size = 4
    num_steps = 1
    batch_size = 1
    draft_token = 0

    logits = np.full(
        (batch_size * (num_steps + 1), vocab_size), -100.0, dtype=np.float32
    )
    logits[:, draft_token] = 100.0
    draft_tokens_np = np.full((batch_size, num_steps), draft_token, np.int64)
    draft_probs_np = np.full((batch_size, num_steps), 10.0, dtype=np.float32)

    saw_reject = False
    for seed in range(1, 9):
        out = model.execute(
            *_sampled_stochastic_sampler_inputs(
                device,
                batch_size,
                vocab_size,
                draft_tokens_np,
                logits,
                np.ones(batch_size, dtype=np.float32),
                draft_probs_np,
                seed=seed,
            )
        )
        first_rejected, recovered, bonus = (
            cast(Buffer, o).to_numpy() for o in out
        )
        assert np.isfinite(first_rejected).all()
        assert np.isfinite(recovered).all()
        assert np.isfinite(bonus).all()
        assert ((recovered >= 0) & (recovered < vocab_size)).all()
        saw_reject |= bool(first_rejected[0] == 0)
    assert saw_reject


def test_stochastic_acceptance_sampler_mixed_per_row_params() -> None:
    """Guards per-row indexing of temperature/top_k/top_p."""
    device = Accelerator()
    session = InferenceSession(devices=[device])
    graph = build_stochastic_acceptance_sampler_graph(
        device=DeviceRef.from_device(device)
    )
    model = session.load(graph)

    vocab_size = 6
    num_steps = 2
    batch_size = 3

    target_argmax = [2, 4, 1]
    draft_tokens_np = np.array([[2, 2], [0, 0], [1, 1]], dtype=np.int64)
    logits = np.full(
        (batch_size * (num_steps + 1), vocab_size), -100.0, dtype=np.float32
    )
    for b, argmax_token in enumerate(target_argmax):
        for s in range(num_steps + 1):
            logits[b * (num_steps + 1) + s, argmax_token] = 100.0

    temperature_np = np.array([0.0, 1.0, 2.5], dtype=np.float32)
    top_k_np = np.array([1, vocab_size, 3], dtype=np.int64)
    top_p_np = np.array([1.0, 0.9, 0.5], dtype=np.float32)

    first_rejected, recovered, bonus = model.execute(
        *_stochastic_sampler_inputs(
            device,
            batch_size,
            vocab_size,
            draft_tokens_np,
            logits,
            temperature_np,
            top_k_np=top_k_np,
            top_p_np=top_p_np,
        )
    )
    first_rejected_np = cast(Buffer, first_rejected).to_numpy()
    np.testing.assert_array_equal(first_rejected_np, [num_steps, 0, num_steps])
    assert np.isfinite(cast(Buffer, recovered).to_numpy()).all()
    assert np.isfinite(cast(Buffer, bonus).to_numpy()).all()


def test_stochastic_acceptance_sampler_greedy_is_seed_independent() -> None:
    """A ``temperature==0`` row reduces to argmax and is seed-independent.

    Uses a one-ULP near-tie -- the regime where the stochastic path's coin and
    Gumbel draws would otherwise flip with the seed.
    """
    device = Accelerator()
    session = InferenceSession(devices=[device])
    graph = build_stochastic_acceptance_sampler_graph(
        device=DeviceRef.from_device(device)
    )
    model = session.load(graph)

    vocab_size = 4
    num_steps = 3
    batch_size = 1

    logits = np.full(
        (batch_size * (num_steps + 1), vocab_size), -100.0, dtype=np.float32
    )
    logits[:, 0] = np.float32(10.0)
    logits[:, 1] = np.nextafter(np.float32(10.0), np.float32(20.0))
    draft_tokens_np = np.zeros((batch_size, num_steps), dtype=np.int64)
    temperature_np = np.zeros(batch_size, dtype=np.float32)

    def _run(seed: int) -> tuple[npt.NDArray[Any], ...]:
        outs = model.execute(
            *_stochastic_sampler_inputs(
                device,
                batch_size,
                vocab_size,
                draft_tokens_np,
                logits,
                temperature_np,
                top_k_np=np.ones(batch_size, dtype=np.int64),
                seed=seed,
            )
        )
        return tuple(cast(Buffer, o).to_numpy().copy() for o in outs)

    base = _run(1)
    for seed in (2, 3):
        for got, expected in zip(_run(seed), base, strict=True):
            np.testing.assert_array_equal(got, expected)

    np.testing.assert_array_equal(base[0], [0])
    np.testing.assert_array_equal(base[1], [[1, 1, 1]])
    np.testing.assert_array_equal(base[2], [[1]])


def test_stochastic_acceptance_sampler_bonus_token_uses_seed() -> None:
    """Bonus token sampling must honor the per-execute seed.

    With num_steps=0 the bonus token is the only varying output, so distinct
    seeds must produce different draws and a repeated seed must reproduce them.
    """
    device = Accelerator()
    session = InferenceSession(devices=[device])
    graph = build_stochastic_acceptance_sampler_graph(
        device=DeviceRef.from_device(device)
    )
    model = session.load(graph)

    # Flat logits + temperature=1.0 give a near-uniform draw, so different
    # seeds diverge.
    batch_size = 16
    vocab_size = 64
    num_steps = 0
    draft_tokens_np = np.zeros((batch_size, num_steps), dtype=np.int64)
    logits = np.zeros((batch_size, vocab_size), dtype=np.float32)
    temperature_np = np.ones(batch_size, dtype=np.float32)

    def _bonus_for_seed(seed: int) -> npt.NDArray[np.int64]:
        _, _, bonus = model.execute(
            *_stochastic_sampler_inputs(
                device,
                batch_size,
                vocab_size,
                draft_tokens_np,
                logits,
                temperature_np,
                seed=seed,
            )
        )
        return cast(Buffer, bonus).to_numpy()

    bonus_seed_1 = _bonus_for_seed(1)
    bonus_seed_2 = _bonus_for_seed(2)
    bonus_seed_1_again = _bonus_for_seed(1)

    assert bonus_seed_1.shape == (batch_size, 1)
    assert ((bonus_seed_1 >= 0) & (bonus_seed_1 < vocab_size)).all()

    assert not np.array_equal(bonus_seed_1, bonus_seed_2)
    np.testing.assert_array_equal(bonus_seed_1, bonus_seed_1_again)


def _build_relaxed_acceptance_graph(
    device_ref: DeviceRef,
    *,
    relaxed_topk: int,
    relaxed_delta: float,
) -> Graph:
    """Inline graph that calls ``stochastic_acceptance_sampler`` with the
    new optional kwargs wired in. ``relaxed_topk`` and ``relaxed_delta``
    bake in at graph build time."""
    input_types = [
        TensorType(DType.int64, ["batch_size", "num_steps"], device=device_ref),
        TensorType(
            DType.float32,
            ["total_output_len", "vocab_size"],
            device=device_ref,
        ),
        TensorType(DType.float32, ["batch_size"], device=device_ref),
        TensorType(DType.int64, ["batch_size"], device=device_ref),
        TensorType(DType.int64, [], device=DeviceRef.CPU()),
        TensorType(DType.float32, ["batch_size"], device=device_ref),
        TensorType(DType.float32, [], device=DeviceRef.CPU()),
        ops.random.SeedType(device_ref),
        TensorType(DType.bool, ["batch_size"], device=device_ref),
    ]
    with Graph("stochastic_relaxed", input_types=input_types) as graph:
        (
            draft_tokens,
            target_logits,
            temperature,
            top_k,
            max_k,
            top_p,
            min_top_p,
            seed,
            in_thinking_phase,
        ) = graph.inputs
        first_rejected_out, recovered_out, bonus_out = (
            stochastic_acceptance_sampler(
                draft_tokens=draft_tokens.tensor,
                target_logits=target_logits.tensor,
                temperature=temperature.tensor,
                top_k=top_k.tensor,
                max_k=max_k.tensor,
                top_p=top_p.tensor,
                min_top_p=min_top_p.tensor,
                seed=seed.tensor,
                in_thinking_phase=in_thinking_phase.tensor,
                relaxed_topk=relaxed_topk,
                relaxed_delta=relaxed_delta,
            )
        )
        graph.output(first_rejected_out, recovered_out, bonus_out)
    return graph


def _relaxed_inputs(
    device: Device,
    batch_size: int,
    vocab_size: int,
    draft_tokens_np: npt.NDArray[np.int64],
    logits: npt.NDArray[np.float32],
    in_thinking_np: npt.NDArray[np.bool_],
) -> list[Buffer]:
    return [
        *_stochastic_sampler_inputs(
            device,
            batch_size,
            vocab_size,
            draft_tokens_np,
            logits,
            np.ones(batch_size, dtype=np.float32),
        ),
        Buffer.from_numpy(in_thinking_np).to(device),
    ]


def test_relaxed_acceptance_accepts_top_n_within_delta() -> None:
    """When the draft token is within the target's top-N candidates and
    its probability exceeds ``top1 - delta``, the relaxed branch must
    accept all draft positions for thinking rows."""
    device = Accelerator()
    session = InferenceSession(devices=[device])
    device_ref = DeviceRef.from_device(device)

    vocab_size = 5
    num_steps = 2
    batch_size = 1

    logits = np.full((num_steps + 1, vocab_size), -100.0, dtype=np.float32)
    logits[:, 0] = 3.0
    logits[:, 1] = 2.0

    draft = np.full((batch_size, num_steps), 1, dtype=np.int64)
    in_thinking = np.array([True], dtype=np.bool_)

    graph = _build_relaxed_acceptance_graph(
        device_ref, relaxed_topk=3, relaxed_delta=0.95
    )
    model = session.load(graph)
    first_rejected, _, _ = model.execute(
        *_relaxed_inputs(
            device, batch_size, vocab_size, draft, logits, in_thinking
        )
    )
    assert int(cast(Buffer, first_rejected).to_numpy()[0]) == num_steps


def test_relaxed_acceptance_rejects_out_of_top_n() -> None:
    """A draft token that is not among the target's top-N candidates
    must be rejected at position 0 even with ``in_thinking_phase=True``."""
    device = Accelerator()
    session = InferenceSession(devices=[device])
    device_ref = DeviceRef.from_device(device)

    vocab_size = 5
    num_steps = 2
    batch_size = 1

    logits = np.full((num_steps + 1, vocab_size), -100.0, dtype=np.float32)
    logits[:, 0] = 5.0
    logits[:, 1] = 4.0
    # Token 4 has logit -100 → not in top-3 → relaxed rejects.
    draft = np.full((batch_size, num_steps), 4, dtype=np.int64)
    in_thinking = np.array([True], dtype=np.bool_)

    graph = _build_relaxed_acceptance_graph(
        device_ref, relaxed_topk=3, relaxed_delta=0.95
    )
    model = session.load(graph)
    first_rejected, _, _ = model.execute(
        *_relaxed_inputs(
            device, batch_size, vocab_size, draft, logits, in_thinking
        )
    )
    assert int(cast(Buffer, first_rejected).to_numpy()[0]) == 0


# Bitmask-constrained acceptance sampling tests.


def test_stochastic_acceptance_sampler_with_bitmask_rejects_invalid_draft() -> (
    None
):
    """Tests that stochastic sampler rejects draft tokens masked out by bitmask.

    When a draft token is masked (bitmask=False), its target probability
    becomes ~0 after softmax, so the draft should be rejected at that position.
    """
    device = Accelerator()
    session = InferenceSession(devices=[device])
    device_ref = DeviceRef.from_device(device)

    vocab_size = 6
    num_steps = 3
    batch_size = 2

    # Build graph with bitmask input
    input_types = [
        TensorType(DType.int64, ["batch_size", "num_steps"], device=device_ref),
        TensorType(
            DType.float32, ["total_output_len", "vocab_size"], device=device_ref
        ),
        TensorType(DType.float32, ["batch_size"], device=device_ref),
        TensorType(DType.int64, ["batch_size"], device=device_ref),
        TensorType(DType.int64, [], device=DeviceRef.CPU()),
        TensorType(DType.float32, ["batch_size"], device=device_ref),
        TensorType(DType.float32, [], device=DeviceRef.CPU()),
        TensorType(
            DType.int32,
            ["batch_size", "num_bitmask_positions", "packed_vocab_size"],
            device=device_ref,
        ),
    ]

    with Graph("stochastic_with_bitmask", input_types=input_types) as graph:
        (
            draft_tokens,
            target_logits,
            temperature,
            top_k,
            max_k,
            top_p,
            min_top_p,
            token_bitmasks,
        ) = graph.inputs
        first_rejected, recovered, bonus = stochastic_acceptance_sampler(
            draft_tokens=draft_tokens.tensor,
            target_logits=target_logits.tensor,
            temperature=temperature.tensor,
            top_k=top_k.tensor,
            max_k=max_k.tensor,
            top_p=top_p.tensor,
            min_top_p=min_top_p.tensor,
            seed=ops.constant(42, dtype=DType.uint64, device=device_ref),
            token_bitmasks=token_bitmasks.tensor,
        )
        graph.output(first_rejected, recovered, bonus)

    model = session.load(graph)

    # Target logits: all tokens have equal logits (uniform after softmax)
    target_logits_np = np.zeros(
        (batch_size * (num_steps + 1), vocab_size), dtype=np.float32
    )

    # Draft tokens: all are token 2
    draft_tokens_np = np.full((batch_size, num_steps), 2, dtype=np.int64)

    # Bitmask: Batch 0 allows token 2 everywhere → should accept all
    # Bitmask: Batch 1 disallows token 2 at position 1 → should reject at position 1
    bitmask_np = np.ones((batch_size, num_steps + 1, vocab_size), dtype=bool)
    bitmask_np[1, 1, 2] = False  # Mask out token 2 at position 1 for batch 1

    result = model.execute(
        Buffer.from_numpy(draft_tokens_np).to(device),
        Buffer.from_numpy(target_logits_np).to(device),
        Buffer.from_numpy(np.ones(batch_size, dtype=np.float32)).to(device),
        Buffer.from_numpy(np.full(batch_size, vocab_size, dtype=np.int64)).to(
            device
        ),
        Buffer.from_numpy(np.array(vocab_size, dtype=np.int64)),
        Buffer.from_numpy(np.ones(batch_size, dtype=np.float32)).to(device),
        Buffer.from_numpy(np.array(1.0, dtype=np.float32)),
        Buffer.from_numpy(_pack_bitmask(bitmask_np)).to(device),
    )

    first_rejected_np = cast(Buffer, result[0]).to_numpy()

    # Batch 0: token 2 is allowed everywhere → may accept (probabilistic)
    # Batch 1: token 2 is masked at position 1 → probability ~0, should reject at 1
    # Note: With uniform logits, acceptance is probabilistic even for allowed tokens.
    # But masked token has probability ~0, so rejection at position 1 is deterministic.
    assert first_rejected_np[1] <= 1, (
        f"Batch 1 should reject at position 1 or earlier, got {first_rejected_np[1]}"
    )


def test_stochastic_acceptance_sampler_bitmask_constrains_recovered_tokens() -> (
    None
):
    """Tests that recovered tokens respect the grammar bitmask constraints.

    When a draft token is rejected, the recovered token should be sampled
    from the valid tokens according to the bitmask.
    """
    device = Accelerator()
    session = InferenceSession(devices=[device])
    device_ref = DeviceRef.from_device(device)

    vocab_size = 6
    num_steps = 2
    batch_size = 1

    input_types = [
        TensorType(DType.int64, ["batch_size", "num_steps"], device=device_ref),
        TensorType(
            DType.float32, ["total_output_len", "vocab_size"], device=device_ref
        ),
        TensorType(DType.float32, ["batch_size"], device=device_ref),
        TensorType(DType.int64, ["batch_size"], device=device_ref),
        TensorType(DType.int64, [], device=DeviceRef.CPU()),
        TensorType(DType.float32, ["batch_size"], device=device_ref),
        TensorType(DType.float32, [], device=DeviceRef.CPU()),
        TensorType(
            DType.int32,
            ["batch_size", "num_bitmask_positions", "packed_vocab_size"],
            device=device_ref,
        ),
    ]

    with Graph(
        "stochastic_recovered_constrained", input_types=input_types
    ) as graph:
        (
            draft_tokens,
            target_logits,
            temperature,
            top_k,
            max_k,
            top_p,
            min_top_p,
            token_bitmasks,
        ) = graph.inputs
        first_rejected, recovered, bonus = stochastic_acceptance_sampler(
            draft_tokens=draft_tokens.tensor,
            target_logits=target_logits.tensor,
            temperature=temperature.tensor,
            top_k=top_k.tensor,
            max_k=max_k.tensor,
            top_p=top_p.tensor,
            min_top_p=min_top_p.tensor,
            seed=ops.constant(42, dtype=DType.uint64, device=device_ref),
            token_bitmasks=token_bitmasks.tensor,
        )
        graph.output(first_rejected, recovered, bonus)

    model = session.load(graph)

    # Target logits: token 5 has highest logit, but will be masked
    # Token 3 has second highest, allowed by bitmask
    target_logits_np = np.zeros(
        (batch_size * (num_steps + 1), vocab_size), dtype=np.float32
    )
    target_logits_np[:, 5] = 100.0  # Highest but will be masked
    target_logits_np[:, 3] = 50.0  # Second highest, allowed

    # Draft tokens force rejection (token 0 which has low probability)
    draft_tokens_np = np.zeros((batch_size, num_steps), dtype=np.int64)

    # Bitmask: only allow tokens 1, 2, 3 (not 0, 4, 5)
    bitmask_np = np.zeros((batch_size, num_steps + 1, vocab_size), dtype=bool)
    bitmask_np[:, :, [1, 2, 3]] = True

    result = model.execute(
        Buffer.from_numpy(draft_tokens_np).to(device),
        Buffer.from_numpy(target_logits_np).to(device),
        Buffer.from_numpy(np.ones(batch_size, dtype=np.float32)).to(device),
        Buffer.from_numpy(np.full(batch_size, vocab_size, dtype=np.int64)).to(
            device
        ),
        Buffer.from_numpy(np.array(vocab_size, dtype=np.int64)),
        Buffer.from_numpy(np.ones(batch_size, dtype=np.float32)).to(device),
        Buffer.from_numpy(np.array(1.0, dtype=np.float32)),
        Buffer.from_numpy(_pack_bitmask(bitmask_np)).to(device),
    )

    recovered_np = cast(Buffer, result[1]).to_numpy()

    # Recovered tokens should be from valid set {1, 2, 3}
    # With token 3 having highest logit among valid tokens, it should be selected
    valid_tokens = {1, 2, 3}
    for pos in range(num_steps):
        assert recovered_np[0, pos] in valid_tokens, (
            f"Recovered token at position {pos} should be in {valid_tokens}, "
            f"got {recovered_np[0, pos]}"
        )


def test_stochastic_acceptance_sampler_bitmask_constrains_bonus_token() -> None:
    """Tests that bonus token respects the grammar bitmask at the final position.

    The bonus token is sampled at position num_steps (the +1 position),
    and should be constrained by the bitmask at that position.
    """
    device = Accelerator()
    session = InferenceSession(devices=[device])
    device_ref = DeviceRef.from_device(device)

    vocab_size = 6
    num_steps = 2
    batch_size = 1

    input_types = [
        TensorType(DType.int64, ["batch_size", "num_steps"], device=device_ref),
        TensorType(
            DType.float32, ["total_output_len", "vocab_size"], device=device_ref
        ),
        TensorType(DType.float32, ["batch_size"], device=device_ref),
        TensorType(DType.int64, ["batch_size"], device=device_ref),
        TensorType(DType.int64, [], device=DeviceRef.CPU()),
        TensorType(DType.float32, ["batch_size"], device=device_ref),
        TensorType(DType.float32, [], device=DeviceRef.CPU()),
        TensorType(
            DType.int32,
            ["batch_size", "num_bitmask_positions", "packed_vocab_size"],
            device=device_ref,
        ),
    ]

    with Graph(
        "stochastic_bonus_constrained", input_types=input_types
    ) as graph:
        (
            draft_tokens,
            target_logits,
            temperature,
            top_k,
            max_k,
            top_p,
            min_top_p,
            token_bitmasks,
        ) = graph.inputs
        first_rejected, recovered, bonus = stochastic_acceptance_sampler(
            draft_tokens=draft_tokens.tensor,
            target_logits=target_logits.tensor,
            temperature=temperature.tensor,
            top_k=top_k.tensor,
            max_k=max_k.tensor,
            top_p=top_p.tensor,
            min_top_p=min_top_p.tensor,
            seed=ops.constant(42, dtype=DType.uint64, device=device_ref),
            token_bitmasks=token_bitmasks.tensor,
        )
        graph.output(first_rejected, recovered, bonus)

    model = session.load(graph)

    # Target logits: token 0 has highest logit at bonus position, but masked
    # Token 4 allowed at bonus position with second highest
    target_logits_np = np.zeros(
        (batch_size * (num_steps + 1), vocab_size), dtype=np.float32
    )
    # Bonus position is last row
    target_logits_np[-1, 0] = 100.0  # Highest but masked
    target_logits_np[-1, 4] = 50.0  # Second highest, allowed

    # Draft tokens that will be accepted (token 1 which has some probability)
    draft_tokens_np = np.full((batch_size, num_steps), 1, dtype=np.int64)
    target_logits_np[:-1, 1] = 100.0  # Make draft tokens likely to accept

    # Bitmask: different constraints at each position
    # Positions 0, 1: allow all tokens
    # Position 2 (bonus): only allow tokens 3, 4, 5
    bitmask_np = np.ones((batch_size, num_steps + 1, vocab_size), dtype=bool)
    bitmask_np[:, -1, [0, 1, 2]] = (
        False  # Mask tokens 0, 1, 2 at bonus position
    )

    result = model.execute(
        Buffer.from_numpy(draft_tokens_np).to(device),
        Buffer.from_numpy(target_logits_np).to(device),
        Buffer.from_numpy(np.ones(batch_size, dtype=np.float32)).to(device),
        Buffer.from_numpy(np.full(batch_size, vocab_size, dtype=np.int64)).to(
            device
        ),
        Buffer.from_numpy(np.array(vocab_size, dtype=np.int64)),
        Buffer.from_numpy(np.ones(batch_size, dtype=np.float32)).to(device),
        Buffer.from_numpy(np.array(1.0, dtype=np.float32)),
        Buffer.from_numpy(_pack_bitmask(bitmask_np)).to(device),
    )

    bonus_np = cast(Buffer, result[2]).to_numpy()

    # Bonus token should be from valid set at bonus position {3, 4, 5}
    # With token 4 having highest logit among valid tokens
    valid_bonus_tokens = {3, 4, 5}
    assert bonus_np[0, 0] in valid_bonus_tokens, (
        f"Bonus token should be in {valid_bonus_tokens}, got {bonus_np[0, 0]}"
    )


def test_stochastic_acceptance_sampler_all_true_bitmask_unchanged_behavior() -> (
    None
):
    """Tests that an all-True bitmask produces same results as no bitmask.

    This validates that the bitmask path doesn't alter behavior for
    unconstrained requests (which use all-True bitmasks).
    """
    device = Accelerator()
    session = InferenceSession(devices=[device])
    device_ref = DeviceRef.from_device(device)

    vocab_size = 6
    num_steps = 3
    batch_size = 2

    # Build graph WITHOUT bitmask
    input_types_no_mask = [
        TensorType(DType.int64, ["batch_size", "num_steps"], device=device_ref),
        TensorType(
            DType.float32, ["total_output_len", "vocab_size"], device=device_ref
        ),
        TensorType(DType.float32, ["batch_size"], device=device_ref),
        TensorType(DType.int64, ["batch_size"], device=device_ref),
        TensorType(DType.int64, [], device=DeviceRef.CPU()),
        TensorType(DType.float32, ["batch_size"], device=device_ref),
        TensorType(DType.float32, [], device=DeviceRef.CPU()),
    ]

    with Graph("stochastic_no_mask", input_types=input_types_no_mask) as graph:
        (
            draft_tokens,
            target_logits,
            temperature,
            top_k,
            max_k,
            top_p,
            min_top_p,
        ) = graph.inputs
        first_rejected, recovered, bonus = stochastic_acceptance_sampler(
            draft_tokens=draft_tokens.tensor,
            target_logits=target_logits.tensor,
            temperature=temperature.tensor,
            top_k=top_k.tensor,
            max_k=max_k.tensor,
            top_p=top_p.tensor,
            min_top_p=min_top_p.tensor,
            seed=ops.constant(42, dtype=DType.uint64, device=device_ref),
            token_bitmasks=None,
        )
        graph.output(first_rejected, recovered, bonus)

    model_no_mask = session.load(graph)

    # Build graph WITH all-True bitmask
    input_types_with_mask = input_types_no_mask + [
        TensorType(
            DType.int32,
            ["batch_size", "num_bitmask_positions", "packed_vocab_size"],
            device=device_ref,
        ),
    ]

    with Graph(
        "stochastic_all_true_mask", input_types=input_types_with_mask
    ) as graph:
        (
            draft_tokens,
            target_logits,
            temperature,
            top_k,
            max_k,
            top_p,
            min_top_p,
            token_bitmasks,
        ) = graph.inputs
        first_rejected, recovered, bonus = stochastic_acceptance_sampler(
            draft_tokens=draft_tokens.tensor,
            target_logits=target_logits.tensor,
            temperature=temperature.tensor,
            top_k=top_k.tensor,
            max_k=max_k.tensor,
            top_p=top_p.tensor,
            min_top_p=min_top_p.tensor,
            seed=ops.constant(42, dtype=DType.uint64, device=device_ref),
            token_bitmasks=token_bitmasks.tensor,
        )
        graph.output(first_rejected, recovered, bonus)

    model_with_mask = session.load(graph)

    # Test data
    target_logits_np = np.zeros(
        (batch_size * (num_steps + 1), vocab_size), dtype=np.float32
    )
    target_logits_np[:, 2] = 100.0  # Token 2 has highest probability

    draft_tokens_np = np.full((batch_size, num_steps), 2, dtype=np.int64)
    all_true_bitmask = np.ones(
        (batch_size, num_steps + 1, vocab_size), dtype=bool
    )

    common_inputs = [
        Buffer.from_numpy(draft_tokens_np).to(device),
        Buffer.from_numpy(target_logits_np).to(device),
        Buffer.from_numpy(np.ones(batch_size, dtype=np.float32)).to(device),
        Buffer.from_numpy(np.full(batch_size, vocab_size, dtype=np.int64)).to(
            device
        ),
        Buffer.from_numpy(np.array(vocab_size, dtype=np.int64)),
        Buffer.from_numpy(np.ones(batch_size, dtype=np.float32)).to(device),
        Buffer.from_numpy(np.array(1.0, dtype=np.float32)),
    ]

    result_no_mask = model_no_mask.execute(*common_inputs)
    result_with_mask = model_with_mask.execute(
        *common_inputs,
        Buffer.from_numpy(_pack_bitmask(all_true_bitmask)).to(device),
    )

    # Results should be identical with same seed
    np.testing.assert_array_equal(
        cast(Buffer, result_no_mask[0]).to_numpy(),
        cast(Buffer, result_with_mask[0]).to_numpy(),
    )
    np.testing.assert_array_equal(
        cast(Buffer, result_no_mask[1]).to_numpy(),
        cast(Buffer, result_with_mask[1]).to_numpy(),
    )
    np.testing.assert_array_equal(
        cast(Buffer, result_no_mask[2]).to_numpy(),
        cast(Buffer, result_with_mask[2]).to_numpy(),
    )


def _load_greedy_bitmask_model(
    session: InferenceSession, device_ref: DeviceRef
) -> Model:
    """Compiles greedy_acceptance_sampler with a bitmask input."""
    input_types = [
        TensorType(DType.int64, ["batch_size", "num_steps"], device=device_ref),
        TensorType(
            DType.float32, ["total_output_len", "vocab_size"], device=device_ref
        ),
        TensorType(
            DType.int32,
            ["batch_size", "num_bitmask_positions", "packed_vocab_size"],
            device=device_ref,
        ),
    ]
    with Graph("greedy_with_bitmask", input_types=input_types) as graph:
        draft_tokens, target_logits, token_bitmasks = graph.inputs
        first_rejected, recovered, bonus = greedy_acceptance_sampler(
            draft_tokens=draft_tokens.tensor,
            target_logits=target_logits.tensor,
            token_bitmasks=token_bitmasks.tensor,
        )
        graph.output(first_rejected, recovered, bonus)
    return session.load(graph)


def test_greedy_acceptance_sampler_bitmask_rejects_and_recovers() -> None:
    """Masked greedy: a draft equal to the UNMASKED argmax but masked out is
    rejected, and recovered/bonus tokens are the masked argmax."""
    device = Accelerator()
    session = InferenceSession(devices=[device])
    model = _load_greedy_bitmask_model(session, DeviceRef.from_device(device))

    batch_size = 2
    num_steps = 3
    vocab_size = 6

    # Unmasked argmax is token 2 at every position; token 4 is runner-up.
    target_logits = np.zeros(
        (batch_size * (num_steps + 1), vocab_size), dtype=np.float32
    )
    target_logits[:, 2] = 10.0
    target_logits[:, 4] = 5.0

    draft_tokens = np.full((batch_size, num_steps), 2, dtype=np.int64)

    # Batch 0: all-allowed -> accepts everything, bonus = 2.
    # Batch 1: token 2 masked out at position 1 -> reject there; the masked
    # argmax (recovered token) at position 1 is the runner-up, token 4.
    bitmask = np.ones((batch_size, num_steps + 1, vocab_size), dtype=bool)
    bitmask[1, 1, 2] = False

    result = model.execute(
        Buffer.from_numpy(draft_tokens).to(device),
        Buffer.from_numpy(target_logits).to(device),
        Buffer.from_numpy(_pack_bitmask(bitmask)).to(device),
    )
    first_rejected = cast(Buffer, result[0]).to_numpy()
    recovered = cast(Buffer, result[1]).to_numpy()
    bonus = cast(Buffer, result[2]).to_numpy()

    np.testing.assert_array_equal(first_rejected, [num_steps, 1])
    np.testing.assert_array_equal(recovered[0], [2, 2, 2])
    np.testing.assert_array_equal(recovered[1], [2, 4, 2])
    np.testing.assert_array_equal(bonus, [[2], [2]])


def test_greedy_acceptance_sampler_bitmask_bonus_token_masked() -> None:
    """The bonus position's mask constrains the bonus token."""
    device = Accelerator()
    session = InferenceSession(devices=[device])
    model = _load_greedy_bitmask_model(session, DeviceRef.from_device(device))

    batch_size = 1
    num_steps = 2
    vocab_size = 6

    target_logits = np.zeros(
        (batch_size * (num_steps + 1), vocab_size), dtype=np.float32
    )
    target_logits[:, 2] = 10.0
    target_logits[:, 3] = 5.0

    draft_tokens = np.full((batch_size, num_steps), 2, dtype=np.int64)

    bitmask = np.ones((batch_size, num_steps + 1, vocab_size), dtype=bool)
    bitmask[0, num_steps, 2] = False  # bonus position forbids the argmax

    result = model.execute(
        Buffer.from_numpy(draft_tokens).to(device),
        Buffer.from_numpy(target_logits).to(device),
        Buffer.from_numpy(_pack_bitmask(bitmask)).to(device),
    )
    np.testing.assert_array_equal(
        cast(Buffer, result[0]).to_numpy(), [num_steps]
    )
    np.testing.assert_array_equal(cast(Buffer, result[2]).to_numpy(), [[3]])


def test_greedy_acceptance_sampler_bitmask_fill_is_hard_exclusion() -> None:
    """A masked token cannot win even when every valid logit is far below
    -10000 (the stochastic path's finite fill) -- the greedy mask fill must
    behave as -inf."""
    device = Accelerator()
    session = InferenceSession(devices=[device])
    model = _load_greedy_bitmask_model(session, DeviceRef.from_device(device))

    batch_size = 1
    num_steps = 1
    vocab_size = 6

    # Every valid logit sits below -10000; the masked token holds the max.
    target_logits = np.full(
        (batch_size * (num_steps + 1), vocab_size), -30000.0, dtype=np.float32
    )
    target_logits[:, 2] = 0.0  # unmasked argmax, but grammar-invalid
    target_logits[:, 4] = -20000.0  # best valid token

    draft_tokens = np.full((batch_size, num_steps), 2, dtype=np.int64)

    bitmask = np.ones((batch_size, num_steps + 1, vocab_size), dtype=bool)
    bitmask[:, :, 2] = False

    result = model.execute(
        Buffer.from_numpy(draft_tokens).to(device),
        Buffer.from_numpy(target_logits).to(device),
        Buffer.from_numpy(_pack_bitmask(bitmask)).to(device),
    )
    np.testing.assert_array_equal(cast(Buffer, result[0]).to_numpy(), [0])
    np.testing.assert_array_equal(cast(Buffer, result[1]).to_numpy(), [[4]])
    np.testing.assert_array_equal(cast(Buffer, result[2]).to_numpy(), [[4]])


def test_greedy_acceptance_sampler_all_allowed_matches_unmasked() -> None:
    """An all-allowed bitmask reproduces the unmasked greedy results."""
    device = Accelerator()
    session = InferenceSession(devices=[device])
    device_ref = DeviceRef.from_device(device)
    masked_model = _load_greedy_bitmask_model(session, device_ref)
    plain_graph = build_greedy_acceptance_sampler_graph(device=device_ref)
    plain_model = session.load(plain_graph)

    batch_size = 3
    num_steps = 4
    vocab_size = 17  # not a multiple of 32: exercises packing padding

    rng = np.random.default_rng(7)
    target_logits = rng.standard_normal(
        (batch_size * (num_steps + 1), vocab_size)
    ).astype(np.float32)
    draft_tokens = rng.integers(
        0, vocab_size, size=(batch_size, num_steps)
    ).astype(np.int64)
    bitmask = np.ones((batch_size, num_steps + 1, vocab_size), dtype=bool)

    plain = plain_model.execute(
        Buffer.from_numpy(draft_tokens).to(device),
        Buffer.from_numpy(target_logits).to(device),
    )
    masked = masked_model.execute(
        Buffer.from_numpy(draft_tokens).to(device),
        Buffer.from_numpy(target_logits).to(device),
        Buffer.from_numpy(_pack_bitmask(bitmask)).to(device),
    )
    for p, m in zip(plain, masked, strict=True):
        np.testing.assert_array_equal(
            cast(Buffer, p).to_numpy(), cast(Buffer, m).to_numpy()
        )


def test_greedy_acceptance_sampler_bitmask_is_graph_capturable() -> None:
    """The masked-greedy op sequence (apply_packed_bitmask + argmax) must
    record into a device graph and replay with masked results -- capturability
    is why greedy acceptance exists, and the stochastic path cannot capture."""
    device = Accelerator()
    if device.api != "cuda":
        pytest.skip("device graph capture smoke is validated on CUDA")
    session = InferenceSession(devices=[device])
    model = _load_greedy_bitmask_model(session, DeviceRef.from_device(device))

    batch_size = 2
    num_steps = 3
    vocab_size = 6

    target_logits = np.zeros(
        (batch_size * (num_steps + 1), vocab_size), dtype=np.float32
    )
    target_logits[:, 2] = 10.0
    target_logits[:, 4] = 5.0
    draft_tokens = np.full((batch_size, num_steps), 2, dtype=np.int64)
    bitmask = np.ones((batch_size, num_steps + 1, vocab_size), dtype=bool)
    bitmask[1, 1, 2] = False

    inputs = [
        Buffer.from_numpy(draft_tokens).to(device),
        Buffer.from_numpy(target_logits).to(device),
        Buffer.from_numpy(_pack_bitmask(bitmask)).to(device),
    ]
    outputs = model.capture(0, *inputs)
    model.replay(0, *inputs)
    device.synchronize()

    np.testing.assert_array_equal(outputs[0].to_numpy(), [num_steps, 1])
    np.testing.assert_array_equal(outputs[1].to_numpy()[1], [2, 4, 2])
    np.testing.assert_array_equal(outputs[2].to_numpy(), [[2], [2]])


def _load_stochastic_model(
    session: InferenceSession, device_ref: DeviceRef, per_row_seed: bool
) -> Model:
    """Compiles stochastic_acceptance_sampler with a rank-0 or rank-1 seed."""
    seed_type = (
        TensorType(DType.uint64, ["batch_size"], device=device_ref)
        if per_row_seed
        else TensorType(DType.uint64, [], device=device_ref)
    )
    input_types = [
        TensorType(DType.int64, ["batch_size", "num_steps"], device=device_ref),
        TensorType(
            DType.float32, ["total_output_len", "vocab_size"], device=device_ref
        ),
        TensorType(DType.float32, ["batch_size"], device=device_ref),
        TensorType(DType.int64, ["batch_size"], device=device_ref),
        TensorType(DType.int64, [], device=DeviceRef.CPU()),
        TensorType(DType.float32, ["batch_size"], device=device_ref),
        TensorType(DType.float32, [], device=DeviceRef.CPU()),
        seed_type,
    ]
    with Graph(
        f"stochastic_seed_rank{int(per_row_seed)}", input_types=input_types
    ) as graph:
        draft, logits, temp, top_k, max_k, top_p, min_top_p, seed = graph.inputs
        outs = stochastic_acceptance_sampler(
            draft_tokens=draft.tensor,
            target_logits=logits.tensor,
            temperature=temp.tensor,
            top_k=top_k.tensor,
            max_k=max_k.tensor,
            top_p=top_p.tensor,
            min_top_p=min_top_p.tensor,
            seed=seed.tensor,
        )
        graph.output(*outs)
    return session.load(graph)


def _run_stochastic(
    model: Model,
    device: Device,
    draft: npt.NDArray[np.int64],
    logits: npt.NDArray[np.float32],
    temps: list[float],
    seeds: npt.NDArray[np.uint64],
    vocab_size: int,
    top_ks: list[int] | None = None,
) -> list[npt.NDArray[np.int64]]:
    batch = draft.shape[0]
    if top_ks is None:
        top_ks = [vocab_size] * batch
    result = model.execute(
        Buffer.from_numpy(draft).to(device),
        Buffer.from_numpy(logits).to(device),
        Buffer.from_numpy(np.asarray(temps, dtype=np.float32)).to(device),
        Buffer.from_numpy(np.asarray(top_ks, dtype=np.int64)).to(device),
        Buffer.from_numpy(np.array(vocab_size, dtype=np.int64)),
        Buffer.from_numpy(np.ones(batch, dtype=np.float32)).to(device),
        Buffer.from_numpy(np.array(1.0, dtype=np.float32)),
        Buffer.from_numpy(seeds).to(device),
    )
    return [cast(Buffer, x).to_numpy() for x in result]


def test_stochastic_per_row_seed_single_row_matches_scalar_seed() -> None:
    """A rank-1 [batch] seed at batch size 1 is bit-identical to the rank-0
    scalar-seed derivation, so migrating a caller cannot change single-row
    sampling."""
    device = Accelerator()
    session = InferenceSession(devices=[device])
    device_ref = DeviceRef.from_device(device)
    m0 = _load_stochastic_model(session, device_ref, per_row_seed=False)
    m1 = _load_stochastic_model(session, device_ref, per_row_seed=True)

    vocab_size = 16
    num_steps = 3
    rng = np.random.default_rng(11)
    for _ in range(20):
        draft = rng.integers(0, vocab_size, size=(1, num_steps)).astype(
            np.int64
        )
        logits = (
            rng.standard_normal((num_steps + 1, vocab_size)).astype(np.float32)
            * 3
        )
        seed = np.uint64(rng.integers(0, 2**62))
        scalar = _run_stochastic(
            m0,
            device,
            draft,
            logits,
            [0.9],
            np.array(seed, dtype=np.uint64),
            vocab_size,
        )
        per_row = _run_stochastic(
            m1,
            device,
            draft,
            logits,
            [0.9],
            np.array([seed], dtype=np.uint64),
            vocab_size,
        )
        for a, b in zip(scalar, per_row, strict=True):
            np.testing.assert_array_equal(a, b)


def test_stochastic_per_row_seed_rows_are_independent_of_coresidents() -> None:
    """With per-row seeds, a row's outputs at a fixed batch position are a
    function of its own (seed, logits, params) only: changing every OTHER
    row's content leaves it untouched, and changing its own seed changes it.
    Under the scalar row-0 seed neither holds."""
    device = Accelerator()
    session = InferenceSession(devices=[device])
    device_ref = DeviceRef.from_device(device)
    model = _load_stochastic_model(session, device_ref, per_row_seed=True)

    vocab_size = 16
    num_steps = 3
    batch = 3
    rng = np.random.default_rng(21)
    draft = rng.integers(0, vocab_size, size=(batch, num_steps)).astype(
        np.int64
    )
    logits = (
        rng.standard_normal((batch * (num_steps + 1), vocab_size)).astype(
            np.float32
        )
        * 3
    )
    seeds = rng.integers(0, 2**62, size=batch).astype(np.uint64)
    temps = [0.7, 1.1, 0.9]
    base = _run_stochastic(
        model, device, draft, logits, temps, seeds, vocab_size
    )

    for _ in range(10):
        draft2, logits2, seeds2, temps2 = (
            draft.copy(),
            logits.copy(),
            seeds.copy(),
            list(temps),
        )
        for other in (0, 2):
            draft2[other] = rng.integers(0, vocab_size, size=num_steps)
            logits2[other * (num_steps + 1) : (other + 1) * (num_steps + 1)] = (
                rng.standard_normal((num_steps + 1, vocab_size)).astype(
                    np.float32
                )
                * 3
            )
            seeds2[other] = rng.integers(0, 2**62)
            temps2[other] = float(rng.uniform(0.5, 1.3))
        out = _run_stochastic(
            model, device, draft2, logits2, temps2, seeds2, vocab_size
        )
        assert out[0][1] == base[0][1]
        np.testing.assert_array_equal(out[1][1], base[1][1])
        np.testing.assert_array_equal(out[2][1], base[2][1])

    reseeded = seeds.copy()
    reseeded[1] += np.uint64(1)
    out = _run_stochastic(
        model, device, draft, logits, temps, reseeded, vocab_size
    )
    assert (
        out[0][1] != base[0][1]
        or (out[1][1] != base[1][1]).any()
        or (out[2][1] != base[2][1]).any()
    ), "row 1's stream must be keyed by its own seed"


def test_stochastic_per_row_seed_honors_topk_at_every_position() -> None:
    """Per-row top_k applies at every verify position: with top_k=1 the
    stochastic path degenerates to argmax, so argmax drafts are all accepted
    and recovered/bonus tokens are the per-position argmax."""
    device = Accelerator()
    session = InferenceSession(devices=[device])
    device_ref = DeviceRef.from_device(device)
    model = _load_stochastic_model(session, device_ref, per_row_seed=True)

    vocab_size = 16
    num_steps = 3
    batch = 2
    rng = np.random.default_rng(31)
    logits = (
        rng.standard_normal((batch * (num_steps + 1), vocab_size)).astype(
            np.float32
        )
        * 3
    )
    expect = logits.reshape(batch, num_steps + 1, vocab_size).argmax(-1)
    draft = expect[:, :num_steps].astype(np.int64)
    seeds = rng.integers(0, 2**62, size=batch).astype(np.uint64)

    out = _run_stochastic(
        model,
        device,
        draft,
        logits,
        [0.8] * batch,
        seeds,
        vocab_size,
        top_ks=[1] * batch,
    )
    np.testing.assert_array_equal(out[0], [num_steps] * batch)
    np.testing.assert_array_equal(out[1], expect[:, :num_steps])
    np.testing.assert_array_equal(out[2][:, 0], expect[:, num_steps])


def test_stochastic_per_row_seed_is_graph_capturable() -> None:
    """The rank-1 seed derivation keeps the acceptance subgraph capturable,
    and a replay reproduces the same seed-keyed samples."""
    device = Accelerator()
    if device.api != "cuda":
        pytest.skip("device graph capture smoke is validated on CUDA")
    session = InferenceSession(devices=[device])
    device_ref = DeviceRef.from_device(device)
    model = _load_stochastic_model(session, device_ref, per_row_seed=True)

    vocab_size = 16
    num_steps = 3
    batch = 2
    rng = np.random.default_rng(41)
    draft = rng.integers(0, vocab_size, size=(batch, num_steps)).astype(
        np.int64
    )
    logits = (
        rng.standard_normal((batch * (num_steps + 1), vocab_size)).astype(
            np.float32
        )
        * 3
    )
    seeds = rng.integers(0, 2**62, size=batch).astype(np.uint64)
    eager = _run_stochastic(
        model, device, draft, logits, [0.8] * batch, seeds, vocab_size
    )

    inputs = [
        Buffer.from_numpy(draft).to(device),
        Buffer.from_numpy(logits).to(device),
        Buffer.from_numpy(np.full(batch, 0.8, dtype=np.float32)).to(device),
        Buffer.from_numpy(np.full(batch, vocab_size, dtype=np.int64)).to(
            device
        ),
        Buffer.from_numpy(np.array(vocab_size, dtype=np.int64)),
        Buffer.from_numpy(np.ones(batch, dtype=np.float32)).to(device),
        Buffer.from_numpy(np.array(1.0, dtype=np.float32)),
        Buffer.from_numpy(seeds).to(device),
    ]
    outputs = model.capture(0, *inputs)
    model.replay(0, *inputs)
    device.synchronize()
    for expected, replayed in zip(eager, outputs, strict=True):
        np.testing.assert_array_equal(expected, replayed.to_numpy())


def test_stochastic_per_row_seed_composes_with_bitmask() -> None:
    """Per-row seeds and the grammar bitmask compose: a masked-out draft is
    rejected and recovered/bonus tokens satisfy the constraint, per row."""
    device = Accelerator()
    session = InferenceSession(devices=[device])
    device_ref = DeviceRef.from_device(device)

    vocab_size = 16
    num_steps = 2
    batch = 2
    input_types = [
        TensorType(DType.int64, ["batch_size", "num_steps"], device=device_ref),
        TensorType(
            DType.float32, ["total_output_len", "vocab_size"], device=device_ref
        ),
        TensorType(DType.float32, ["batch_size"], device=device_ref),
        TensorType(DType.int64, ["batch_size"], device=device_ref),
        TensorType(DType.int64, [], device=DeviceRef.CPU()),
        TensorType(DType.float32, ["batch_size"], device=device_ref),
        TensorType(DType.float32, [], device=DeviceRef.CPU()),
        TensorType(DType.uint64, ["batch_size"], device=device_ref),
        TensorType(
            DType.int32,
            ["batch_size", "num_bitmask_positions", "packed_vocab_size"],
            device=device_ref,
        ),
    ]
    with Graph("stochastic_perrow_bitmask", input_types=input_types) as graph:
        (draft, logits, temp, top_k, max_k, top_p, min_top_p, seed, bitmask) = (
            graph.inputs
        )
        outs = stochastic_acceptance_sampler(
            draft_tokens=draft.tensor,
            target_logits=logits.tensor,
            temperature=temp.tensor,
            top_k=top_k.tensor,
            max_k=max_k.tensor,
            top_p=top_p.tensor,
            min_top_p=min_top_p.tensor,
            seed=seed.tensor,
            token_bitmasks=bitmask.tensor,
        )
        graph.output(*outs)
    model = session.load(graph)

    # Token 2 dominates every position; batch 1 masks it out at position 0,
    # where only token 4 remains legal.
    logits_np = np.full(
        (batch * (num_steps + 1), vocab_size), -10.0, dtype=np.float32
    )
    logits_np[:, 2] = 10.0
    logits_np[:, 4] = 5.0
    draft_np = np.full((batch, num_steps), 2, dtype=np.int64)
    bitmask_np = np.ones((batch, num_steps + 1, vocab_size), dtype=bool)
    bitmask_np[1, 0, :] = False
    bitmask_np[1, 0, 4] = True
    seeds_np = np.array([7, 8], dtype=np.uint64)

    result = model.execute(
        Buffer.from_numpy(draft_np).to(device),
        Buffer.from_numpy(logits_np).to(device),
        Buffer.from_numpy(np.full(batch, 0.8, dtype=np.float32)).to(device),
        Buffer.from_numpy(np.full(batch, vocab_size, dtype=np.int64)).to(
            device
        ),
        Buffer.from_numpy(np.array(vocab_size, dtype=np.int64)),
        Buffer.from_numpy(np.ones(batch, dtype=np.float32)).to(device),
        Buffer.from_numpy(np.array(1.0, dtype=np.float32)),
        Buffer.from_numpy(seeds_np).to(device),
        Buffer.from_numpy(_pack_bitmask(bitmask_np)).to(device),
    )
    first_rejected = cast(Buffer, result[0]).to_numpy()
    recovered = cast(Buffer, result[1]).to_numpy()

    # Batch 0 (all-allowed): token 2 dominates, drafts accepted.
    assert first_rejected[0] == num_steps
    # Batch 1: the masked-out draft at position 0 is rejected and the
    # recovered token is the only legal one.
    assert first_rejected[1] == 0
    assert recovered[1][0] == 4


def test_stochastic_seedtype_input_keeps_shared_seed_semantics() -> None:
    """A rank-1 ``[1]`` seed is the graph-level SeedType input — one seed for
    the whole batch, whatever the batch size — and must keep the scalar-seed
    behavior bit-for-bit rather than being read as a per-row tensor (the
    shape algebra cannot even unify ``[1]`` with a symbolic batch)."""
    device = Accelerator()
    session = InferenceSession(devices=[device])
    device_ref = DeviceRef.from_device(device)
    scalar_model = _load_stochastic_model(
        session, device_ref, per_row_seed=False
    )

    input_types = [
        TensorType(DType.int64, ["batch_size", "num_steps"], device=device_ref),
        TensorType(
            DType.float32, ["total_output_len", "vocab_size"], device=device_ref
        ),
        TensorType(DType.float32, ["batch_size"], device=device_ref),
        TensorType(DType.int64, ["batch_size"], device=device_ref),
        TensorType(DType.int64, [], device=DeviceRef.CPU()),
        TensorType(DType.float32, ["batch_size"], device=device_ref),
        TensorType(DType.float32, [], device=DeviceRef.CPU()),
        ops.random.SeedType(device_ref),
    ]
    with Graph("stochastic_seedtype", input_types=input_types) as graph:
        draft, logits, temp, top_k, max_k, top_p, min_top_p, seed = graph.inputs
        outs = stochastic_acceptance_sampler(
            draft_tokens=draft.tensor,
            target_logits=logits.tensor,
            temperature=temp.tensor,
            top_k=top_k.tensor,
            max_k=max_k.tensor,
            top_p=top_p.tensor,
            min_top_p=min_top_p.tensor,
            seed=seed.tensor,
        )
        graph.output(*outs)
    seedtype_model = session.load(graph)

    vocab_size = 16
    num_steps = 3
    batch = 4  # larger than the seed's [1]: shared-seed semantics
    rng = np.random.default_rng(51)
    draft_np = rng.integers(0, vocab_size, size=(batch, num_steps)).astype(
        np.int64
    )
    logits_np = (
        rng.standard_normal((batch * (num_steps + 1), vocab_size)).astype(
            np.float32
        )
        * 3
    )
    seed_np = np.uint64(rng.integers(0, 2**62))

    shared = seedtype_model.execute(
        Buffer.from_numpy(draft_np).to(device),
        Buffer.from_numpy(logits_np).to(device),
        Buffer.from_numpy(np.full(batch, 0.9, dtype=np.float32)).to(device),
        Buffer.from_numpy(np.full(batch, vocab_size, dtype=np.int64)).to(
            device
        ),
        Buffer.from_numpy(np.array(vocab_size, dtype=np.int64)),
        Buffer.from_numpy(np.ones(batch, dtype=np.float32)).to(device),
        Buffer.from_numpy(np.array(1.0, dtype=np.float32)),
        Buffer.from_numpy(np.array([seed_np], dtype=np.uint64)).to(device),
    )
    scalar = _run_stochastic(
        scalar_model,
        device,
        draft_np,
        logits_np,
        [0.9] * batch,
        np.array(seed_np, dtype=np.uint64),
        vocab_size,
    )
    for a, b in zip(shared, scalar, strict=True):
        np.testing.assert_array_equal(cast(Buffer, a).to_numpy(), b)


def test_stochastic_per_row_seed_accepts_foreign_batch_dim() -> None:
    """A per-row seed input whose batch dim carries a different symbolic name
    still builds and runs: the sampler rebinds it to the canonical batch dim
    before the flat derivation."""
    device = Accelerator()
    session = InferenceSession(devices=[device])
    device_ref = DeviceRef.from_device(device)

    input_types = [
        TensorType(DType.int64, ["batch_size", "num_steps"], device=device_ref),
        TensorType(
            DType.float32, ["total_output_len", "vocab_size"], device=device_ref
        ),
        TensorType(DType.float32, ["batch_size"], device=device_ref),
        TensorType(DType.int64, ["batch_size"], device=device_ref),
        TensorType(DType.int64, [], device=DeviceRef.CPU()),
        TensorType(DType.float32, ["batch_size"], device=device_ref),
        TensorType(DType.float32, [], device=DeviceRef.CPU()),
        TensorType(DType.uint64, ["seed_rows"], device=device_ref),
    ]
    with Graph("stochastic_foreign_dim", input_types=input_types) as graph:
        draft, logits, temp, top_k, max_k, top_p, min_top_p, seed = graph.inputs
        outs = stochastic_acceptance_sampler(
            draft_tokens=draft.tensor,
            target_logits=logits.tensor,
            temperature=temp.tensor,
            top_k=top_k.tensor,
            max_k=max_k.tensor,
            top_p=top_p.tensor,
            min_top_p=min_top_p.tensor,
            seed=seed.tensor,
        )
        graph.output(*outs)
    model = session.load(graph)

    vocab_size = 16
    num_steps = 3
    batch = 3
    rng = np.random.default_rng(61)
    draft_np = rng.integers(0, vocab_size, size=(batch, num_steps)).astype(
        np.int64
    )
    logits_np = (
        rng.standard_normal((batch * (num_steps + 1), vocab_size)).astype(
            np.float32
        )
        * 3
    )
    seeds_np = rng.integers(0, 2**62, size=batch).astype(np.uint64)
    result = model.execute(
        Buffer.from_numpy(draft_np).to(device),
        Buffer.from_numpy(logits_np).to(device),
        Buffer.from_numpy(np.full(batch, 0.9, dtype=np.float32)).to(device),
        Buffer.from_numpy(np.full(batch, vocab_size, dtype=np.int64)).to(
            device
        ),
        Buffer.from_numpy(np.array(vocab_size, dtype=np.int64)),
        Buffer.from_numpy(np.ones(batch, dtype=np.float32)).to(device),
        Buffer.from_numpy(np.array(1.0, dtype=np.float32)),
        Buffer.from_numpy(seeds_np).to(device),
    )
    assert cast(Buffer, result[0]).to_numpy().shape == (batch,)
