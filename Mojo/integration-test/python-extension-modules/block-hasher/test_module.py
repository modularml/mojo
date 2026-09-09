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

import hashlib
import time
import timeit

# Imports from 'mojo_module.so'
import mojo_module  # type: ignore[import-not-found]
import numpy as np


def s2us(s: float) -> float:
    return s * 1000 * 1000


# see: https://github.com/modularml/modular/blob/851943b46d2a38b36883009f42fa669ee7d41a2c/SDK/lib/API/python/max/nn/kv_cache/kv_cache/block_utils.py#L76
def naive_python_hashing(tokens: np.ndarray, block_size: int) -> list[int]:
    num_elts = tokens.size
    num_hashes = num_elts // block_size

    # Initial hash seed value
    prev_hash = 0

    results = []
    for i in range(num_hashes):
        block = tokens[i * block_size : (i + 1) * block_size]
        pair_to_hash = (prev_hash, tuple(block))
        curr_hash = hash(pair_to_hash)
        results.append(curr_hash)
        prev_hash = curr_hash

    return results


# https://github.com/vllm-project/vllm/blob/a5450f11c95847cf51a17207af9a3ca5ab569b2c/vllm/distributed/kv_transfer/kv_connector/mooncake_store_connector.py#L196
def python_tensor_hash(tokens: np.ndarray, block_size: int) -> list[int]:
    """Calculate the hash value of the tensor."""
    num_elts = tokens.size
    num_hashes = num_elts // block_size

    # Initial hash seed value
    prev_hash = 0

    results = []
    for i in range(num_hashes):
        block = tokens[i * block_size : (i + 1) * block_size]
        digest = hashlib.blake2b(block.tobytes()).hexdigest()
        pair_to_hash = (prev_hash, digest)
        curr_hash = hash(pair_to_hash)

        results.append(curr_hash)

    return results


def print_bench_run(
    name: str,
    callable,  # noqa: ANN001
    *,
    iter_count: int = 1000,
) -> None:
    total_time = timeit.timeit(callable, number=iter_count)

    print(
        "=== Bench:",
        name,
        "was",
        f"{s2us(total_time / iter_count):.2f}",
        "µs per iteration",
    )

    # Call it once more so the user can compare the results.
    print("\tresult = ", repr(callable()))


def test_block_hasher() -> None:
    print("-" * 80)
    print("Hello from Block Hasher Example!")
    print("-" * 80)

    block_size = 128
    tokens = np.arange(3000, dtype=np.int32)  # Use int32 for tokens

    time.sleep(1)

    print_bench_run(
        "Mojo 🔥",
        lambda: mojo_module.mojo_block_hasher(tokens, block_size),
        iter_count=10_000,
    )

    time.sleep(0.1)

    print_bench_run(
        "Python Tensor Hash",
        lambda: python_tensor_hash(tokens, block_size),
        iter_count=10_000,
    )

    time.sleep(0.1)

    print_bench_run(
        "Python Naive",
        lambda: naive_python_hashing(tokens, block_size),
        iter_count=10_000,
    )
