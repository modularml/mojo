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

import numpy as np
import pytest
from max.pipelines.kv_cache.paged_kv_cache.block_utils import (
    hash_request_tokens,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("block_size", [1, 2, 4, 64, 128, 256, 1024])
@pytest.mark.parametrize("prompt_len", [16, 65536])
async def test_basic(block_size: int, prompt_len: int) -> None:
    prompt = np.arange(prompt_len, dtype=np.int32)
    hash_vals = hash_request_tokens(prompt, block_size)
    assert len(hash_vals) == prompt_len // block_size

    # Check that they form a chain
    for i in range(1, len(hash_vals)):
        block_hash = hash_vals[i]
        block_token_ids = prompt[i * block_size : (i + 1) * block_size]
        expected_hash = block_hash
        parent_hash_value = hash_vals[i - 1]
        actual_hash = hash_request_tokens(
            block_token_ids,
            block_size,
            parent_hash_value,
        )[0]
        assert expected_hash == actual_hash

    # Check that the hash values are non-zero.
    # Technically a 0 hash is possible, but it's extremely unlikely and usually
    # indicates a bug in the hasher.
    assert b"\x00" * 8 not in hash_vals

    # Check that the hash values are unique.
    assert len(set(hash_vals)) == len(hash_vals)


def check_for_collisions(
    prompt_1: np.ndarray, prompt_2: np.ndarray, block_size: int
) -> None:
    hash_vals_1 = hash_request_tokens(prompt_1, block_size, b"\x00" * 8)
    hash_vals_2 = hash_request_tokens(prompt_2, block_size, b"\x00" * 8)

    for i, x in enumerate(hash_vals_1):
        if x in hash_vals_2:
            j = hash_vals_2.index(x)
            raise ValueError(
                f"Collision found at idx={i} and idx={j} with hash value {x!r}"
            )

    # There should be no duplicate hashes.
    assert len(hash_vals_1) + len(hash_vals_2) == len(
        set(hash_vals_1) | set(hash_vals_2)
    )


@pytest.mark.asyncio
async def test_collision() -> None:
    block_size = 1

    prompt_1 = np.array([1, 1, 0, 1, 0, 0, 0, 1, 1, 0])
    prompt_2 = np.array([0, 1, 0, 1, 0, 0, 0, 1, 1, 0])

    check_for_collisions(prompt_1, prompt_2, block_size)


@pytest.mark.asyncio
@pytest.mark.parametrize("block_size", [1, 128])
async def test_collision_random(block_size: int) -> None:
    # Picking too large of number of iterations can cause test to timeout.
    iterations = 10

    for _ in range(iterations):
        # Generate arbitrary suffix
        random_prompt = np.random.randint(0, 2, size=8192)

        # Append either 0 or 1 to the beginning of the prompt so they result in
        # unique hashes,
        prompt_1 = np.concatenate([np.array([0]), random_prompt])
        prompt_2 = np.concatenate([np.array([1]), random_prompt])

        # Check for collisions
        check_for_collisions(prompt_1, prompt_2, block_size)
