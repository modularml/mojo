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

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.engine import InferenceSession, Model
from max.graph import DeviceRef
from max.pipelines.context import LogProbabilities
from max.pipelines.lib.log_probabilities import (
    compute_log_probabilities_ragged,
    log_probabilities_ragged_graph,
)
from test_common.numerics import log_softmax


def _check_log_probabilities_equal(
    actual: LogProbabilities, expected: LogProbabilities
) -> None:
    np.testing.assert_allclose(
        actual.token_log_probabilities, expected.token_log_probabilities
    )
    assert len(actual.top_log_probabilities) == len(
        expected.top_log_probabilities
    )
    for actual_top, expected_top in zip(
        actual.top_log_probabilities,
        expected.top_log_probabilities,
        strict=True,
    ):
        assert actual_top.keys() == expected_top.keys()
        key_order = list(expected_top.keys())
        actual_values = [actual_top[key] for key in key_order]
        expected_values = [expected_top[key] for key in key_order]
        np.testing.assert_allclose(actual_values, expected_values, rtol=1e-6)


@pytest.fixture(scope="module")
def cpu_device() -> CPU:
    return CPU()


@pytest.fixture(scope="module")
def cpu_session(cpu_device: CPU) -> InferenceSession:
    return InferenceSession(devices=[cpu_device])


@pytest.fixture(scope="module")
def cpu_model(cpu_device: CPU, cpu_session: InferenceSession) -> Model:
    graph = log_probabilities_ragged_graph(
        DeviceRef.from_device(cpu_device), levels=3
    )
    return cpu_session.load(graph)


def test_compute_log_probabilities(cpu_device: CPU, cpu_model: Model) -> None:
    device = cpu_device
    model = cpu_model

    input_row_offsets = np.array([0, 2], dtype=np.uint32)
    batch_logits = np.array(
        [
            [0.5, 0.25, 0.7, 0.3, 1, 0.05],  # top-3 index = 0, 2, 4
            [0.1, 0.2, 0.9, 0.3, 0.4, 0.14],  # top-3 index = 2, 3, 4
        ],
        dtype=np.float32,
    )
    batch_tokens = np.array([0, 1, 4])
    batch_echo = [True]  # Value doesn't matter

    log_probs = log_softmax(batch_logits, axis=-1)

    # Check top 3
    output = compute_log_probabilities_ragged(
        device,
        model,
        input_row_offsets=input_row_offsets,
        logits=Buffer.from_numpy(batch_logits).to(device),
        next_token_logits=Buffer.from_numpy(batch_logits[-1:]).to(device),
        tokens=batch_tokens[:-1],
        sampled_tokens=batch_tokens[-1:],
        batch_top_n=[3],
        batch_echo=batch_echo,
    )
    assert len(output) == 1
    expected_log_probs = LogProbabilities(
        token_log_probabilities=[log_probs[0][1], log_probs[1][4]],
        top_log_probabilities=[
            {
                0: log_probs[0][0].item(),
                2: log_probs[0][2].item(),
                4: log_probs[0][4].item(),
                # While not part of the top 3, token 1 was sampled so it gets
                # included in the top log probabilities.
                1: log_probs[0][1].item(),
            },
            {
                2: log_probs[1][2].item(),
                3: log_probs[1][3].item(),
                4: log_probs[1][4].item(),
            },
        ],
    )
    assert isinstance(output[0], LogProbabilities)
    _check_log_probabilities_equal(output[0], expected_log_probs)

    # Check top 1
    output = compute_log_probabilities_ragged(
        device,
        model,
        input_row_offsets=input_row_offsets,
        logits=Buffer.from_numpy(batch_logits).to(device),
        next_token_logits=Buffer.from_numpy(batch_logits[-1:]).to(device),
        tokens=batch_tokens[:-1],
        sampled_tokens=batch_tokens[-1:],
        batch_top_n=[1],
        batch_echo=batch_echo,
    )
    assert len(output) == 1
    expected_log_probs = LogProbabilities(
        token_log_probabilities=[log_probs[0][1], log_probs[1][4]],
        top_log_probabilities=[
            {
                4: log_probs[0][4].item(),
                # While not part of the top 1, token 1 was sampled so it is
                # included in the top log probabilities.
                1: log_probs[0][1].item(),
            },
            {
                2: log_probs[1][2].item(),
                # Same case here, token 4 was sampled so it is included.
                4: log_probs[1][4].item(),
            },
        ],
    )
    assert isinstance(output[0], LogProbabilities)
    _check_log_probabilities_equal(output[0], expected_log_probs)


def test_compute_log_probabilities_batch(
    cpu_device: CPU, cpu_model: Model
) -> None:
    device = cpu_device
    model = cpu_model

    input_row_offsets = np.array([0, 2, 3, 4], dtype=np.uint32)
    batch_logits = np.array(
        [
            [0.5, 0.25, 0.7, 0.3, 1, 0.05],  # batch 1 token 1; top index: 4
            [0.1, 0.2, 0.9, 0.3, 0.4, 0.14],  # batch 1 token 2; top index: 2
            [0.4, 0.42, 0.3, 0.89, 0.07, 0.5],  # b. 2; top 5 idx: 0, 1, 2, 3, 5
            [100, 100, 100, 100, 100, 100],  # batch 3
        ],
        dtype=np.float32,
    )
    batch_next_token_logits = np.array(
        [
            [0.1, 0.2, 0.9, 0.3, 0.4, 0.14],  # batch 1; top index: 2
            [0.4, 0.42, 0.3, 0.89, 0.07, 0.5],  # b. 2; top 5 idx: 0, 1, 2, 3, 5
            [100, 100, 100, 100, 100, 100],  # batch 3
        ],
        dtype=np.float32,
    )
    # batch 1 = [0, 1]; batch 2 = [0]; batch 3 = [0]
    batch_tokens = np.array([0, 1, 0, 0])
    batch_sampled_tokens = np.array([4, 3, 3])
    batch_top_n = [1, 5, 0]
    batch_echo = [True, False, True]

    log_probs1 = log_softmax(batch_logits[0:2], axis=-1)
    log_probs2 = log_softmax(batch_logits[2:3], axis=-1)

    output = compute_log_probabilities_ragged(
        device,
        model,
        input_row_offsets=input_row_offsets,
        logits=Buffer.from_numpy(batch_logits).to(device),
        next_token_logits=Buffer.from_numpy(batch_next_token_logits).to(device),
        tokens=batch_tokens,
        sampled_tokens=batch_sampled_tokens,
        batch_top_n=batch_top_n,
        batch_echo=batch_echo,
    )
    assert len(output) == 3
    assert output[2] is None  # Top N was 0, so output[2] should be None.

    assert output[0] is not None
    _check_log_probabilities_equal(
        output[0],
        LogProbabilities(
            token_log_probabilities=[log_probs1[0][1], log_probs1[1][4]],
            top_log_probabilities=[
                {
                    4: log_probs1[0][4].item(),
                    # While not part of the top 1, token 1 was sampled so it is
                    # included in the top log probabilities.
                    1: log_probs1[0][1].item(),
                },
                {
                    2: log_probs1[1][2].item(),
                    # Same case here, token 4 was sampled so it is included.
                    4: log_probs1[1][4].item(),
                },
            ],
        ),
    )
    assert output[1] is not None
    _check_log_probabilities_equal(
        output[1],
        LogProbabilities(
            token_log_probabilities=[log_probs2[0][3]],
            top_log_probabilities=[
                {
                    0: log_probs2[0][0].item(),
                    1: log_probs2[0][1].item(),
                    2: log_probs2[0][2].item(),
                    3: log_probs2[0][3].item(),
                    5: log_probs2[0][5].item(),
                },
            ],
        ),
    )


def test_compute_log_probabilities_ragged(
    cpu_device: CPU, cpu_model: Model
) -> None:
    device = cpu_device
    model = cpu_model

    output = compute_log_probabilities_ragged(
        device,
        model,
        input_row_offsets=np.array([0, 3, 5, 8]),
        logits=Buffer.from_numpy(
            np.array(
                [
                    [10, 11],  # batch 0 token 0
                    [11, 12],  # batch 0 token 1
                    [12, 13],  # batch 0 token 2
                    [20, 21],  # batch 1 token 0
                    [21, 22],  # batch 1 token 1
                    [30, 31],  # batch 2 token 0
                    [31, 32],  # batch 2 token 1
                    [32, 33],  # batch 2 token 2
                ],
                dtype=np.float32,
            )
        ).to(device),
        next_token_logits=Buffer.from_numpy(
            np.array(
                [
                    [12, 13],  # batch 0 token 2
                    [21, 22],  # batch 1 token 1
                    [32, 33],  # batch 2 token 2
                ],
                dtype=np.float32,
            )
        ).to(device),
        tokens=np.array([1, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32),
        sampled_tokens=np.array([0, 1, 0], dtype=np.int32),
        batch_top_n=[1, 0, 1],
        batch_echo=[True, False, False],
    )
    assert len(output) == 3
    assert output[0] is not None
    assert len(output[0].token_log_probabilities) == 3
    assert len(output[0].top_log_probabilities) == 3
    assert output[0].top_log_probabilities[0].keys() == {1}
    assert output[0].top_log_probabilities[1].keys() == {0, 1}
    assert output[0].top_log_probabilities[2].keys() == {0, 1}
    assert output[1] is None
    assert output[2] is not None
    assert len(output[2].token_log_probabilities) == 1
    assert len(output[2].top_log_probabilities) == 1
    assert output[2].top_log_probabilities[0].keys() == {0, 1}


@dataclass
class InputBatchItem:
    logits: np.ndarray | None  # shape (seq, vocab_size)
    next_token_logits: np.ndarray  # shape (vocab_size,)
    tokens: np.ndarray  # shape (seq,)
    sampled_token: int
    top_n: int
    echo: bool

    @classmethod
    def random(
        cls, rng: np.random.Generator, *, vocab_size: int, with_logits: bool
    ) -> InputBatchItem:
        seq_len = int(rng.integers(1, 300, endpoint=True))
        want_log_probs = bool(rng.integers(0, 4) == 0)
        fake_quantize = bool(rng.integers(0, 2) == 0)
        if with_logits:
            # This is absolutely the wrong distribution, but it's good enough
            # for our purposes.  That said, if someone felt like figuring out a
            # more representative distribution for logit outputs and could
            # replace this, the test could have more fidelity.
            logits = rng.normal(size=(seq_len, vocab_size)).astype(np.float32)
            if fake_quantize:
                logits = np.round(logits, decimals=1)
            next_token_logits = logits[-1, :]
        else:
            logits = None
            next_token_logits = rng.normal(size=(vocab_size,)).astype(
                np.float32
            )
            if fake_quantize:
                next_token_logits = np.round(next_token_logits, decimals=1)
        return cls(
            logits=logits,
            next_token_logits=next_token_logits,
            tokens=rng.integers(0, vocab_size, size=(seq_len,)),
            sampled_token=int(rng.integers(0, vocab_size)),
            top_n=(
                int(rng.integers(1, min(5, vocab_size), endpoint=True))
                if want_log_probs
                else 0
            ),
            echo=(
                bool(rng.integers(0, 4) == 0)
                if want_log_probs and with_logits
                else False
            ),
        )


def random_batch(rng: np.random.Generator) -> Sequence[InputBatchItem]:
    batch_size = int(rng.integers(1, 64, endpoint=True))
    vocab_size = int(rng.integers(1, 128, endpoint=True))
    with_logits = bool(rng.integers(0, 4) == 0)
    return [
        InputBatchItem.random(
            rng, vocab_size=vocab_size, with_logits=with_logits
        )
        for i in range(batch_size)
    ]


@dataclass
class PackedInput:
    input_row_offsets: np.ndarray
    logits: np.ndarray | None
    next_token_logits: np.ndarray
    tokens: np.ndarray
    sampled_tokens: np.ndarray
    batch_top_n: Sequence[int]
    batch_echo: Sequence[bool]

    @classmethod
    def from_items(cls, batch_items: Sequence[InputBatchItem]) -> PackedInput:
        assert len(batch_items) >= 1
        if batch_items[0].logits is not None:
            logits_parts = []
            for item in batch_items:
                assert item.logits is not None
                logits_parts.append(item.logits)
            logits = np.concatenate(logits_parts, axis=0)
        else:
            assert all(item.logits is None for item in batch_items)
            logits = None
        return cls(
            input_row_offsets=np.concatenate(
                [
                    np.array([0], dtype=np.uint32),
                    np.cumsum(
                        np.array(
                            [item.tokens.shape[0] for item in batch_items],
                            dtype=np.uint32,
                        )
                    ),
                ]
            ),
            logits=logits,
            next_token_logits=np.stack(
                [item.next_token_logits for item in batch_items]
            ),
            tokens=np.concatenate([item.tokens for item in batch_items]),
            sampled_tokens=np.array(
                [item.sampled_token for item in batch_items], dtype=np.uint32
            ),
            batch_top_n=[item.top_n for item in batch_items],
            batch_echo=[item.echo for item in batch_items],
        )


def verify_position(
    alleged_logprob: float,
    top_mapping: Mapping[int, float],
    *,
    top_n: int,
    sampled: int,
    logits: np.ndarray,
) -> None:
    logsoftmaxed_logits = log_softmax(logits)
    threshold = 1e-4
    # Verify logit values -- keys are checked later.
    assert np.isclose(
        alleged_logprob, logsoftmaxed_logits[sampled], rtol=0, atol=threshold
    )
    for token in top_mapping:
        assert np.isclose(
            top_mapping[token],
            logsoftmaxed_logits[token],
            rtol=0,
            atol=threshold,
        )
    # Sampled token must always appear in top_mapping, even if it would cause
    # us to exceed top_n by 1.
    assert sampled in top_mapping
    assert top_n <= len(top_mapping) <= top_n + 1
    if len(top_mapping) == top_n + 1:
        # The only time we should exceed top_n is when the sampled token was
        # not in top_n.  So in this case, the sampled token had better have the
        # minimum logit value.  If there is some other element with a lower
        # logit value, that's a problem.  Relative orderings in top_mapping
        # should be exact, so no usage of tolerance here.
        assert not any(
            value < top_mapping[sampled] for value in top_mapping.values()
        )
    if 1 < len(top_mapping) < len(logits):
        # Sampled token aside, the items in top_mapping had better really be
        # the top_n.  top_n is not necessarily unique (e.g. if there are tokens
        # with identical logits, which does happen in practice), so we're
        # basically saying "everything in top n must be at least as big as
        # everything not in it".  Threshold isn't needed since we're comparing
        # like-for-like -- this test's computed log-softmax rather than
        # top_mapping's values (those we checked separately earlier).
        min_top = min(
            logsoftmaxed_logits[token]
            for token in top_mapping
            if token != sampled
        )
        max_not_in_mapping = max(
            float(value)
            for index, value in enumerate(logsoftmaxed_logits)
            if index not in top_mapping
        )
        assert max_not_in_mapping <= min_top


def verify_output(
    item: InputBatchItem, output: LogProbabilities | None
) -> None:
    if item.top_n == 0:
        assert output is None
        return
    assert output is not None
    if item.echo:
        assert len(output.token_log_probabilities) == item.tokens.shape[0]
        assert len(output.top_log_probabilities) == item.tokens.shape[0]
        assert item.logits is not None
        for seq in range(item.tokens.shape[0] - 1):
            verify_position(
                output.token_log_probabilities[seq],
                output.top_log_probabilities[seq],
                top_n=item.top_n,
                sampled=item.tokens[seq + 1],
                logits=item.logits[seq, :],
            )
    else:
        assert len(output.token_log_probabilities) == 1
        assert len(output.top_log_probabilities) == 1
    verify_position(
        output.token_log_probabilities[-1],
        output.top_log_probabilities[-1],
        top_n=item.top_n,
        sampled=item.sampled_token,
        logits=item.next_token_logits,
    )


# The randomized test provides larger inputs than the manually-written tests
# above, and handles some ambiguous cases for which there are multiple valid
# answers.  (Specifically, ties in the top-n are implementation-dependent, and
# we want to tolerate any tie-breaking strategy.)
@pytest.mark.parametrize("seed", range(50))
def test_log_probabilities_randomized(
    seed: int, cpu_device: CPU, cpu_model: Model
) -> None:
    device = cpu_device
    model = cpu_model
    rng = np.random.default_rng(seed)
    batch = random_batch(rng)
    packed = PackedInput.from_items(batch)
    outputs = compute_log_probabilities_ragged(
        device=device,
        model=model,
        input_row_offsets=packed.input_row_offsets,
        logits=(
            Buffer.from_numpy(packed.logits).to(device)
            if packed.logits is not None
            else None
        ),
        next_token_logits=(
            Buffer.from_numpy(packed.next_token_logits).to(device)
        ),
        tokens=packed.tokens,
        sampled_tokens=packed.sampled_tokens,
        batch_top_n=packed.batch_top_n,
        batch_echo=packed.batch_echo,
    )
    assert len(outputs) == len(batch)
    for item, output in zip(batch, outputs, strict=True):
        verify_output(item, output)


@pytest.fixture(scope="module")
def gpu_device() -> Accelerator:
    return Accelerator()


@pytest.fixture(scope="module")
def gpu_session(gpu_device: Accelerator) -> InferenceSession:
    return InferenceSession(devices=[gpu_device])


@pytest.fixture(scope="module")
def gpu_model(gpu_device: Accelerator, gpu_session: InferenceSession) -> Model:
    graph = log_probabilities_ragged_graph(
        DeviceRef.from_device(gpu_device), levels=3
    )
    return gpu_session.load(graph)


@pytest.mark.skipif(accelerator_count() == 0, reason="no GPU")
@pytest.mark.parametrize("seed", range(50))
def test_log_probabilities_randomized_gpu(
    seed: int, gpu_device: Accelerator, gpu_model: Model
) -> None:
    device = gpu_device
    model = gpu_model
    rng = np.random.default_rng(seed)
    batch = random_batch(rng)
    packed = PackedInput.from_items(batch)
    outputs = compute_log_probabilities_ragged(
        device=device,
        model=model,
        input_row_offsets=packed.input_row_offsets,
        logits=(
            Buffer.from_numpy(packed.logits).to(device)
            if packed.logits is not None
            else None
        ),
        next_token_logits=(
            Buffer.from_numpy(packed.next_token_logits).to(device)
        ),
        tokens=packed.tokens,
        sampled_tokens=packed.sampled_tokens,
        batch_top_n=packed.batch_top_n,
        batch_echo=packed.batch_echo,
    )
    assert len(outputs) == len(batch)
    for item, output in zip(batch, outputs, strict=True):
        verify_output(item, output)


# ============================================================================
# Tests for log probabilities without full logits (echo=False mode)
# These tests verify the fix for computing log probs when enable_echo is False
# ============================================================================


def test_compute_log_probabilities_without_full_logits(
    cpu_device: CPU, cpu_model: Model
) -> None:
    """Test log probs computation when logits=None (no echo mode).

    When enable_echo=False, the model only returns next_token_logits (last token),
    not full logits for all tokens. The compute_log_probabilities_ragged function
    should handle this by using next_token_logits directly.
    """
    device = cpu_device
    model = cpu_model

    # Simulate a batch of 2 sequences, each with different lengths
    input_row_offsets = np.array([0, 3, 5], dtype=np.uint32)

    # Only next_token_logits are available (no full logits)
    next_token_logits = np.array(
        [
            [0.5, 0.25, 0.7, 0.3, 1.0, 0.05],  # batch 0
            [0.1, 0.2, 0.9, 0.3, 0.4, 0.14],  # batch 1
        ],
        dtype=np.float32,
    )

    # Tokens for each sequence (not used when echo=False, but required by API)
    tokens = np.array([0, 1, 2, 0, 1], dtype=np.int32)
    sampled_tokens = np.array([4, 2], dtype=np.int32)

    # Both sequences have echo=False
    batch_echo = [False, False]
    batch_top_n = [3, 2]

    log_probs = log_softmax(next_token_logits, axis=-1)

    # Call with logits=None to simulate no-echo mode
    output = compute_log_probabilities_ragged(
        device,
        model,
        input_row_offsets=input_row_offsets,
        logits=None,  # Key: no full logits available
        next_token_logits=Buffer.from_numpy(next_token_logits).to(device),
        tokens=tokens,
        sampled_tokens=sampled_tokens,
        batch_top_n=batch_top_n,
        batch_echo=batch_echo,
    )

    assert len(output) == 2

    # Verify batch 0: should have 1 log prob entry (just the sampled token)
    assert output[0] is not None
    assert len(output[0].token_log_probabilities) == 1
    assert len(output[0].top_log_probabilities) == 1
    # The sampled token was 4, so log prob should match
    np.testing.assert_allclose(
        output[0].token_log_probabilities[0],
        log_probs[0][4],
        rtol=1e-5,
    )
    # Top 3 should include tokens with highest log probs
    assert 4 in output[0].top_log_probabilities[0]  # token 4 (sampled)

    # Verify batch 1: should have 1 log prob entry (just the sampled token)
    assert output[1] is not None
    assert len(output[1].token_log_probabilities) == 1
    assert len(output[1].top_log_probabilities) == 1
    # The sampled token was 2, so log prob should match
    np.testing.assert_allclose(
        output[1].token_log_probabilities[0],
        log_probs[1][2],
        rtol=1e-5,
    )
    # Top 2 should include the sampled token
    assert 2 in output[1].top_log_probabilities[0]


def test_compute_log_probabilities_mixed_echo_without_logits_fails(
    cpu_device: CPU, cpu_model: Model
) -> None:
    """Test that requesting echo=True without logits raises an assertion error.

    When logits=None but batch_echo contains True, it should fail because
    we can't compute log probs for input tokens without full logits.
    """
    device = cpu_device
    model = cpu_model

    input_row_offsets = np.array([0, 2], dtype=np.uint32)
    next_token_logits = np.array([[0.5, 0.25, 0.7]], dtype=np.float32)
    tokens = np.array([0, 1], dtype=np.int32)
    sampled_tokens = np.array([2], dtype=np.int32)

    # Request echo=True but don't provide logits
    batch_echo = [True]
    batch_top_n = [1]

    with pytest.raises(AssertionError):
        compute_log_probabilities_ragged(
            device,
            model,
            input_row_offsets=input_row_offsets,
            logits=None,  # No logits
            next_token_logits=Buffer.from_numpy(next_token_logits).to(device),
            tokens=tokens,
            sampled_tokens=sampled_tokens,
            batch_top_n=batch_top_n,
            batch_echo=batch_echo,  # But echo=True
        )


def test_compute_log_probabilities_batch_mixed_top_n_no_echo(
    cpu_device: CPU, cpu_model: Model
) -> None:
    """Test batch with different top_n values and no echo.

    This tests the common case where different requests in a batch have
    different logprobs settings, but none request echo.
    """
    device = cpu_device
    model = cpu_model

    input_row_offsets = np.array([0, 2, 5, 6], dtype=np.uint32)
    next_token_logits = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],  # batch 0: top is token 3
            [4.0, 3.0, 2.0, 1.0],  # batch 1: top is token 0
            [1.0, 1.0, 1.0, 1.0],  # batch 2: all equal
        ],
        dtype=np.float32,
    )
    tokens = np.array([0, 1, 0, 1, 2, 0], dtype=np.int32)
    sampled_tokens = np.array([3, 0, 1], dtype=np.int32)

    # Different top_n for each batch item
    batch_top_n = [2, 0, 3]  # batch 1 doesn't want log probs
    batch_echo = [False, False, False]

    output = compute_log_probabilities_ragged(
        device,
        model,
        input_row_offsets=input_row_offsets,
        logits=None,
        next_token_logits=Buffer.from_numpy(next_token_logits).to(device),
        tokens=tokens,
        sampled_tokens=sampled_tokens,
        batch_top_n=batch_top_n,
        batch_echo=batch_echo,
    )

    assert len(output) == 3

    # Batch 0: requested top_n=2
    assert output[0] is not None
    assert len(output[0].token_log_probabilities) == 1
    assert 3 in output[0].top_log_probabilities[0]  # sampled token

    # Batch 1: requested top_n=0 (no log probs)
    assert output[1] is None

    # Batch 2: requested top_n=3
    assert output[2] is not None
    assert len(output[2].token_log_probabilities) == 1
    assert 1 in output[2].top_log_probabilities[0]  # sampled token
