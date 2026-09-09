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
"""``DFlash2GroupedConv`` against the DFlash2 authors' scalar reference.

:func:`reference_grouped_conv` is a transcription of the expected-value loop
in ``tests/v1/spec_decode/test_dflash2.py::test_grouped_conv_matches_reference``
from the authors' vLLM fork. It is the cheapest exact oracle for the tap
indexing and the block-boundary zeroing, so it is checked before the module is
wired into a layer.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weights import WeightData
from max.pipelines.architectures.dflash2_qwen3_5 import DFlash2GroupedConv

HIDDEN = 8
GROUP_SIZE = 2
NUM_GROUPS = HIDDEN // GROUP_SIZE
BATCH = 3


def reference_grouped_conv(
    hidden_states: np.ndarray,
    delta: np.ndarray,
    base: np.ndarray,
    block_size: int,
    taps: int,
) -> np.ndarray:
    """``out[t] = sum_tap (base[tap] + delta[t, tap, group(c)]) * x[t - tap]``.

    Args:
        hidden_states: ``[batch * block_size, hidden]``.
        delta: ``[batch * block_size, taps, num_groups]``.
        base: ``[taps, hidden]`` for one side.
        block_size: Rows per block; ``t`` is taken block-local.
        taps: Convolution width.

    Returns:
        ``[batch * block_size, hidden]``.
    """
    batch = hidden_states.shape[0] // block_size
    blocks = hidden_states.reshape(batch, block_size, NUM_GROUPS, GROUP_SIZE)
    base_g = base.reshape(taps, NUM_GROUPS, GROUP_SIZE)
    delta_g = delta.reshape(batch, block_size, taps, NUM_GROUPS)
    out = np.zeros_like(blocks)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            out[:, position] += (
                base_g[tap] + delta_g[:, position, tap, :, None]
            ) * blocks[:, position - tap]
    return out.reshape(batch * block_size, HIDDEN)


def _fixture(block_size: int, taps: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    tokens = BATCH * block_size
    shape = (tokens, HIDDEN)
    return {
        "x": rng.standard_normal(shape, dtype=np.float32),
        "sublayer_out": rng.standard_normal(shape, dtype=np.float32),
        "base_kernel": rng.standard_normal((2, taps, HIDDEN), dtype=np.float32),
        "projection": rng.standard_normal(
            (2 * taps * NUM_GROUPS, HIDDEN), dtype=np.float32
        ),
    }


def _run_max(
    fixture: dict[str, np.ndarray], block_size: int, taps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Returns ``(prepare_output, finish_output)`` from the MAX module."""
    conv = DFlash2GroupedConv(
        HIDDEN,
        taps=taps,
        group_size=GROUP_SIZE,
        block_size=block_size,
        dtype=DType.float32,
        device=DeviceRef.CPU(),
    )
    conv.load_state_dict(
        {
            "base_kernel": WeightData.from_numpy(
                fixture["base_kernel"], "base_kernel"
            ),
            "kernel_projection.weight": WeightData.from_numpy(
                fixture["projection"], "kernel_projection.weight"
            ),
        },
        strict=True,
    )

    tokens = BATCH * block_size
    token_type = TensorType(DType.float32, [tokens, HIDDEN], DeviceRef.CPU())
    with Graph(
        "dflash2_grouped_conv", input_types=(token_type, token_type)
    ) as graph:
        x, sublayer_out = (value.tensor for value in graph.inputs)
        prepared, coefficients = conv.prepare(x)
        graph.output(prepared, conv.finish(sublayer_out, coefficients))

    session = InferenceSession(devices=[CPU()])
    model = session.load(graph, weights_registry=conv.state_dict())
    outputs = model.execute(
        Buffer.from_numpy(fixture["x"]),
        Buffer.from_numpy(fixture["sublayer_out"]),
    )
    prepared_out, finished_out = outputs
    return np.from_dlpack(prepared_out), np.from_dlpack(finished_out)


def _reference(
    fixture: dict[str, np.ndarray], block_size: int, taps: int
) -> tuple[np.ndarray, np.ndarray]:
    """The scalar reference for both sides, sharing one projection pass."""
    tokens = BATCH * block_size
    coefficients = (fixture["x"] @ fixture["projection"].T).reshape(
        tokens, 2, taps, NUM_GROUPS
    )
    prepared = reference_grouped_conv(
        fixture["x"],
        coefficients[:, 0],
        fixture["base_kernel"][0],
        block_size,
        taps,
    )
    finished = reference_grouped_conv(
        fixture["sublayer_out"],
        coefficients[:, 1],
        fixture["base_kernel"][1],
        block_size,
        taps,
    )
    return prepared, finished


@pytest.mark.parametrize("block_size", [5, 8])
@pytest.mark.parametrize("taps", [1, 2])
def test_grouped_conv_matches_reference(block_size: int, taps: int) -> None:
    fixture = _fixture(block_size, taps, seed=block_size * 10 + taps)
    max_prepared, max_finished = _run_max(fixture, block_size, taps)
    ref_prepared, ref_finished = _reference(fixture, block_size, taps)

    np.testing.assert_allclose(max_prepared, ref_prepared, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(max_finished, ref_finished, rtol=1e-5, atol=1e-5)


def test_conv_reads_only_the_current_and_previous_row_in_block() -> None:
    """A cone-of-influence check, independent of the scalar reference.

    Zeroing the ``t - 1`` tap at block-local position 0 is the whole reason
    the block axis is explicit, and a convolution that wrapped within the
    block or ran across the boundary would still produce plausible, finite
    output. Perturbing the last row of block 0 must therefore move exactly
    that row: not row 0 of the same block (a wrap), and nothing in block 1
    (a boundary crossing).
    """
    block_size, taps = 8, 2
    fixture = _fixture(block_size, taps, seed=7)
    baseline, _ = _run_max(fixture, block_size, taps)

    perturbed = {**fixture, "x": fixture["x"].copy()}
    perturbed["x"][block_size - 1] += 1.0
    after, _ = _run_max(perturbed, block_size, taps)

    assert not np.allclose(baseline[block_size - 1], after[block_size - 1])
    np.testing.assert_array_equal(
        baseline[: block_size - 1], after[: block_size - 1]
    )
    np.testing.assert_array_equal(baseline[block_size:], after[block_size:])


@pytest.mark.parametrize(
    "sabotage",
    ["cross_block_tap", "swapped_sides", "transposed_delta"],
)
def test_sabotaged_references_are_rejected(sabotage: str) -> None:
    """The comparison must fail when the reference is deliberately broken.

    Without this, a passing :func:`test_grouped_conv_matches_reference` would
    only prove the two implementations agree on something, not that they agree
    on the convolution DFlash2 specifies.
    """
    block_size, taps = 8, 2
    fixture = _fixture(block_size, taps, seed=11)
    max_prepared, max_finished = _run_max(fixture, block_size, taps)
    tokens = BATCH * block_size
    coefficients = (fixture["x"] @ fixture["projection"].T).reshape(
        tokens, 2, taps, NUM_GROUPS
    )

    if sabotage == "cross_block_tap":
        # Shift over the flat token axis instead of the block axis, so the
        # t-1 tap of every block's row 0 reads the previous block's last row.
        blocks = fixture["x"].reshape(tokens, NUM_GROUPS, GROUP_SIZE)
        base = fixture["base_kernel"][0].reshape(taps, NUM_GROUPS, GROUP_SIZE)
        coef = base + coefficients[:, 0][..., None]
        broken = coef[:, 0] * blocks
        broken[1:] += coef[1:, 1] * blocks[:-1]
        candidate = broken.reshape(tokens, HIDDEN)
        actual = max_prepared
    elif sabotage == "swapped_sides":
        candidate = reference_grouped_conv(
            fixture["x"],
            coefficients[:, 0],
            fixture["base_kernel"][1],
            block_size,
            taps,
        )
        actual = max_prepared
    else:
        candidate = reference_grouped_conv(
            fixture["sublayer_out"],
            np.ascontiguousarray(
                coefficients[:, 1]
                .reshape(tokens, NUM_GROUPS, taps)
                .transpose(0, 2, 1)
            ),
            fixture["base_kernel"][1],
            block_size,
            taps,
        )
        actual = max_finished

    assert not np.allclose(actual, candidate, rtol=1e-5, atol=1e-5), (
        f"sabotage '{sabotage}' still matched: the check measures nothing"
    )
