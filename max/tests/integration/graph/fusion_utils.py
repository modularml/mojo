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

"""Shared helper for e2e tests that need to verify fusion structure.

Most e2e graph tests only need numeric correctness -- lit tests already cover
fusion structure at the IR level. But structure and behavior can drift apart:
a test can keep computing the right answer after a fusion regression, since
the unfused fallback is still correct, just slower or differently shaped. This
helper is for the cases worth pinning to an actual fusion outcome directly
from Python, rather than trusting that the lit suite alone would catch a
divergence between the two.
"""

from __future__ import annotations

import re

import numpy as np
from max.driver import Buffer
from max.engine import InferenceSession
from max.graph import Graph


def run_and_verify_fusion(
    session: InferenceSession,
    graph: Graph,
    *inputs: np.ndarray,
    fused: str = "",
) -> tuple[np.ndarray, ...]:
    """Compiles ``graph``, checks its fusion structure, then executes it.

    Fusion is a compile-time property: it shows up in
    ``model.kernel_summaries`` right after ``session.load``, so this checks it
    there, before spending time on ``execute``.

    Args:
        session: The session to load ``graph`` with.
        graph: The graph to compile and run.
        *inputs: Executed positional inputs, matching ``graph``'s own input
            order.
        fused: A regular expression checked against ``model.kernel_summaries``
            with ``re.search``. For example, ``"mo.matmul.*mo.relu"`` matches
            a summary listing both leaves in that order, however other leaves
            or fusion-group syntax appear around them. Defaults to ``""``,
            which matches any non-empty summary -- for callers that only need
            to execute the graph, with no fusion claim to check.

    Returns:
        ``graph``'s outputs, each converted to a NumPy array.

    Raises:
        AssertionError: If no summary in ``model.kernel_summaries`` matches
            ``fused``.
    """
    model = session.load(graph)
    if not any(re.search(fused, summary) for summary in model.kernel_summaries):
        raise AssertionError(
            f"expected a kernel summary matching {fused!r}, got: "
            f"{model.kernel_summaries}"
        )

    # A 0-d bool array (a scalar `mo.if`/`ops.cond` predicate) is passed
    # through as a plain Python bool: `Buffer.from_numpy` can't wrap a rank-0
    # bool array (its dtype-view path indexes `shape[-1]`, which doesn't
    # exist for rank 0).
    device_inputs = [
        bool(arr)
        if arr.dtype == np.bool_ and arr.shape == ()
        else Buffer.from_numpy(arr).to(model.input_devices[i])
        for i, arr in enumerate(inputs)
    ]
    results = model.execute(*device_inputs)
    return tuple(r.to_numpy() for r in results)
