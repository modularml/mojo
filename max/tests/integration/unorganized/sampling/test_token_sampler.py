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
"""Tests for token_sampler (basic sampling without structured output)."""

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef
from max.pipelines.context import SamplingParams
from max.pipelines.lib import SamplingConfig, token_sampler


@pytest.fixture(scope="module")
def basic_token_sampler(session: InferenceSession) -> Model:
    """Compile token_sampler once for the module (no structured output)."""
    device_ref = DeviceRef.from_device(session.devices[0])
    sampling_config = SamplingConfig(
        in_dtype=DType.float32,
        out_dtype=DType.float32,
    )
    graph = token_sampler(sampling_config, device=device_ref)
    return session.load(graph)


def test_sampling_with_seed(
    session: InferenceSession, basic_token_sampler: Model
) -> None:
    """Test that sampling with the same seed produces deterministic results."""
    device = session.devices[0]

    # Test parameters
    batch_size = 1
    vocab_size = 1024
    top_k = 100
    seed_1 = 42
    seed_2 = 41
    num_trials = 10

    sampling_params = SamplingParams(top_k=top_k, seed=seed_1)

    temperature = Buffer.from_numpy(
        np.array([sampling_params.temperature] * batch_size, dtype=np.float32)
    ).to(device)
    top_k_np = np.array([sampling_params.top_k] * batch_size, dtype=np.int64)
    top_k_tensor = Buffer.from_numpy(top_k_np).to(device)
    max_k = Buffer.from_numpy(np.array(np.max(top_k_np), dtype=np.int64))
    top_p = Buffer.from_numpy(
        np.array([sampling_params.top_p] * batch_size, dtype=np.float32)
    ).to(device)
    min_top_p = Buffer.from_numpy(
        np.array(sampling_params.top_p, dtype=np.float32)
    )
    min_p = Buffer.from_numpy(
        np.array([0.0] * batch_size, dtype=np.float32)
    ).to(device)
    seed = Buffer.from_numpy(
        np.array([sampling_params.seed] * batch_size, dtype=np.uint64)
    ).to(device)

    # Create a random logits vector [1, vocab_size]
    np.random.seed(123)
    logits_np = np.random.randn(batch_size, vocab_size).astype(np.float32)

    prev_tokens = Buffer(
        shape=(batch_size, 0),
        dtype=DType.int64,
        device=device,
    )
    logits_tensor = Buffer.from_dlpack(logits_np).to(device)

    # Test 1: Sample multiple times with seed=42, results should be identical
    for _ in range(num_trials):
        _tokens, new_prev_tokens = basic_token_sampler(
            logits_tensor,
            prev_tokens,
            top_k_tensor,
            max_k,
            temperature,
            top_p,
            min_top_p,
            min_p,
            seed,
        )[:2]

        assert isinstance(new_prev_tokens, Buffer)
        prev_tokens = new_prev_tokens

    prev_tokens_np = prev_tokens.to_numpy()
    # Verify all results with seed=42 are identical
    for i in range(1, num_trials):
        np.testing.assert_array_equal(
            prev_tokens_np[:, 0],
            prev_tokens_np[:, i],
            err_msg=f"Sampling with seed={seed_1} should be deterministic across runs",
        )

    # Test 2: Sample with seed=41, result should be different from seed=42
    sampling_params_41 = SamplingParams(top_k=top_k, seed=seed_2)
    seed_41 = Buffer.from_numpy(
        np.array([sampling_params_41.seed] * batch_size, dtype=np.uint64)
    ).to(device)
    tokens_41, _ = basic_token_sampler(
        logits_tensor,
        prev_tokens,
        top_k_tensor,
        max_k,
        temperature,
        top_p,
        min_top_p,
        min_p,
        seed_41,
    )[:2]

    assert isinstance(tokens_41, Buffer)
    result_seed_41 = tokens_41.to_numpy()

    assert not np.array_equal(prev_tokens_np[:, 0], result_seed_41), (
        f"Sampling with different seeds ({seed_1} vs {seed_2}) should produce different results"
    )


def test_batch_sampling_arguments(
    session: InferenceSession, basic_token_sampler: Model
) -> None:
    device = session.devices[0]

    # Test parameters
    batch_size = 4
    vocab_size = 8

    prev_tokens = Buffer(
        shape=(batch_size, 0),
        dtype=DType.int64,
        device=device,
    )
    min_p = Buffer.from_numpy(
        np.array([0.0] * batch_size, dtype=np.float32)
    ).to(device)

    num_trials = 100

    def test_top_p_sampling() -> None:
        """Test that different top_p values affect sampling correctly."""
        k = np.array([vocab_size] * batch_size, dtype=np.int64)
        temperature = np.array([1.0] * batch_size, dtype=np.float32)
        logits_np = np.array(
            [[1.0, 0.999, 0.998, 0.997, 0.996, 0.995, 0.994, 0.993]],
            dtype=np.float32,
        )
        batch_logits_np = np.repeat(logits_np, repeats=batch_size, axis=0)
        logits = Buffer.from_dlpack(batch_logits_np).to(device)
        top_k = Buffer.from_numpy(k).to(device)
        max_k = Buffer.from_numpy(np.array(np.max(k), dtype=np.int64))
        temperature_tensor = Buffer.from_numpy(temperature).to(device)

        top_p = np.array([0.51, 0.5, 0.5, 0.5], dtype=np.float32)
        top_p_tensor = Buffer.from_numpy(top_p).to(device)
        min_top_p_tensor = Buffer.from_numpy(
            np.array(np.min(top_p), dtype=np.float32)
        )
        batch_sampled_tokens: list[list[int]] = [[] for _ in range(batch_size)]
        for seed_val in range(num_trials):
            seed_array = np.array(
                [seed_val, seed_val, seed_val, 0], dtype=np.uint64
            )
            seed_tensor = Buffer.from_numpy(seed_array).to(device)
            tokens = basic_token_sampler(
                logits,
                prev_tokens,
                top_k,
                max_k,
                temperature_tensor,
                top_p_tensor,
                min_top_p_tensor,
                min_p,
                seed_tensor,
            )[0]
            assert isinstance(tokens, Buffer)
            tokens_np = tokens.to_numpy()
            for i in range(batch_size):
                batch_sampled_tokens[i].append(tokens_np[i].item())

        # The 0th batch index has a p of 0.51 so it samples from 5 tokens
        expected_tokens = [{0, 1, 2, 3, 4}, {0, 1, 2, 3}, {0, 1, 2, 3}]
        for i in range(batch_size - 1):
            assert set(batch_sampled_tokens[i]) == expected_tokens[i]

        # Seed doesn't change, so it only samples 1 token
        assert len(set(batch_sampled_tokens[batch_size - 1])) == 1

    def test_top_k_sampling() -> None:
        """Test that different top_k values affect sampling correctly."""
        logits_np = np.array(
            [[10.0, 8.0, 2.0, -1.0, -1.5, -2.0, -2.5, -3.0]],
            dtype=np.float32,
        )
        batch_logits_np = np.repeat(logits_np, repeats=batch_size, axis=0)
        logits = Buffer.from_dlpack(batch_logits_np).to(device)
        top_p = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        top_p_tensor = Buffer.from_numpy(top_p).to(device)
        min_top_p_tensor = Buffer.from_numpy(
            np.array(np.min(top_p), dtype=np.float32)
        )

        temperature = np.array([1.0] * batch_size, dtype=np.float32)
        temperature_tensor = Buffer.from_numpy(temperature).to(device)
        batch_sampled_tokens: list[list[int]] = [[] for _ in range(batch_size)]
        for seed_val in range(num_trials):
            seed_array = np.array([seed_val] * batch_size, dtype=np.uint64)
            seed_tensor = Buffer.from_numpy(seed_array).to(device)
            k = np.array([1, 2, 3, 4], dtype=np.int64)
            top_k = Buffer.from_numpy(k).to(device)
            max_k = Buffer.from_numpy(np.array(np.max(k), dtype=np.int64))
            tokens = basic_token_sampler(
                logits,
                prev_tokens,
                top_k,
                max_k,
                temperature_tensor,
                top_p_tensor,
                min_top_p_tensor,
                min_p,
                seed_tensor,
            )[0]
            assert isinstance(tokens, Buffer)
            tokens_np = tokens.to_numpy()
            for i in range(batch_size):
                batch_sampled_tokens[i].append(tokens_np[i].item())

        assert set(batch_sampled_tokens[0]) == {0}
        for i in range(1, batch_size):
            assert len(set(batch_sampled_tokens[i])) > 1

    def test_temperature_sampling() -> None:
        """Test that different temperature values affect sampling correctly."""
        logits_np = np.array(
            [[10.0, 8.0, 2.0, -1.0, -1.5, -2.0, -2.5, -3.0]],
            dtype=np.float32,
        )
        batch_logits_np = np.repeat(logits_np, repeats=batch_size, axis=0)
        logits = Buffer.from_dlpack(batch_logits_np).to(device)
        top_p = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        top_p_tensor = Buffer.from_numpy(top_p).to(device)
        min_top_p_tensor = Buffer.from_numpy(
            np.array(np.min(top_p), dtype=np.float32)
        )
        k = np.array([vocab_size] * batch_size, dtype=np.int64)
        top_k = Buffer.from_numpy(k).to(device)
        max_k = Buffer.from_numpy(np.array(np.max(k), dtype=np.int64))

        batch_sampled_tokens: list[list[int]] = [[] for _ in range(batch_size)]
        for seed_val in range(num_trials):
            seed_array = np.array([seed_val] * batch_size, dtype=np.uint64)
            seed_tensor = Buffer.from_numpy(seed_array).to(device)

            temperature = np.array([0.01, 5.0, 5.0, 5.0], dtype=np.float32)
            temperature_tensor = Buffer.from_numpy(temperature).to(device)

            tokens = basic_token_sampler(
                logits,
                prev_tokens,
                top_k,
                max_k,
                temperature_tensor,
                top_p_tensor,
                min_top_p_tensor,
                min_p,
                seed_tensor,
            )[0]
            assert isinstance(tokens, Buffer)
            tokens_np = tokens.to_numpy()
            for i in range(batch_size):
                batch_sampled_tokens[i].append(tokens_np[i].item())

        # low temperature is more or less deterministic, high temperature is more random
        assert set(batch_sampled_tokens[0]) == {0}
        for i in range(1, batch_size):
            assert len(set(batch_sampled_tokens[i])) > vocab_size // 2

    # Run all tests
    test_top_p_sampling()
    test_top_k_sampling()
    test_temperature_sampling()


def test_top_k_zero_selects_all(
    session: InferenceSession, basic_token_sampler: Model
) -> None:
    """Test that a raw ``top_k=0`` selects all tokens (does not emit ``-1``).

    At the user/graph layer ``topk_fused_sampling`` remaps ``top_k=0`` to ``-1``
    via ``ops.where`` (the kernel's "select all" encoding, in turn remapped to
    ``max_k``). The result must be a valid token id in ``[0, vocab_size)`` and
    never the ``-1`` sentinel. The ``top_k`` buffer is built with a raw ``0``
    (not via ``SamplingParams``, whose ``__post_init__`` would rewrite ``0`` to
    ``-1`` before the graph ever sees it).
    """
    device = session.devices[0]

    batch_size = 2
    vocab_size = 8

    # Peaked logits: with top_k=1 the control row deterministically samples
    # token 0, while the select-all (top_k=0) row is free to sample across the
    # full distribution.
    logits_np = np.array(
        [[10.0, 8.0, 2.0, -1.0, -1.5, -2.0, -2.5, -3.0]],
        dtype=np.float32,
    )
    batch_logits_np = np.repeat(logits_np, repeats=batch_size, axis=0)
    logits = Buffer.from_dlpack(batch_logits_np).to(device)

    prev_tokens = Buffer(
        shape=(batch_size, 0),
        dtype=DType.int64,
        device=device,
    )

    # Row 0: raw top_k=0 (select-all under test). Row 1: top_k=1 control.
    k = np.array([0, 1], dtype=np.int64)
    top_k = Buffer.from_numpy(k).to(device)
    # max_k must be > 0; use vocab_size so the select-all row can see every
    # token.
    max_k = Buffer.from_numpy(np.array(vocab_size, dtype=np.int64))

    temperature = np.array([1.0] * batch_size, dtype=np.float32)
    temperature_tensor = Buffer.from_numpy(temperature).to(device)
    top_p = np.array([1.0] * batch_size, dtype=np.float32)
    top_p_tensor = Buffer.from_numpy(top_p).to(device)
    min_top_p_tensor = Buffer.from_numpy(
        np.array(np.min(top_p), dtype=np.float32)
    )
    min_p = Buffer.from_numpy(
        np.array([0.0] * batch_size, dtype=np.float32)
    ).to(device)

    num_trials = 100
    batch_sampled_tokens: list[list[int]] = [[] for _ in range(batch_size)]
    for seed_val in range(num_trials):
        seed_array = np.array([seed_val] * batch_size, dtype=np.uint64)
        seed_tensor = Buffer.from_numpy(seed_array).to(device)
        tokens = basic_token_sampler(
            logits,
            prev_tokens,
            top_k,
            max_k,
            temperature_tensor,
            top_p_tensor,
            min_top_p_tensor,
            min_p,
            seed_tensor,
        )[0]
        assert isinstance(tokens, Buffer)
        tokens_np = tokens.to_numpy()
        for i in range(batch_size):
            batch_sampled_tokens[i].append(tokens_np[i].item())

    # Primary assertion: the top_k=0 (select-all) row must always produce a
    # valid token id and never the -1 sentinel.
    for tok in batch_sampled_tokens[0]:
        assert tok != -1, "top_k=0 must select-all, not emit the -1 sentinel"
        assert 0 <= tok < vocab_size, (
            f"top_k=0 sampled an out-of-range token {tok}"
        )

    # Stronger assertion: select-all really samples the full distribution
    # rather than just the argmax, so multiple distinct tokens appear.
    assert len(set(batch_sampled_tokens[0])) > 1, (
        "top_k=0 should sample across the full distribution"
    )

    # Control row: top_k=1 deterministically samples the argmax (token 0).
    assert set(batch_sampled_tokens[1]) == {0}
