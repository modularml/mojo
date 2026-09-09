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
"""Tests for graph compilation and runtime error messages.

Verifies that error messages from graph compilation failures and runtime
errors include proper Python source tracebacks and useful context.

Environment variables required (set via BUILD.bazel env):
  MODULAR_DEBUG=source-tracebacks  -- enables Python stack traces in errors
  MODULAR_DEVICE_CONTEXT_SYNC_MODE=1 -- synchronous GPU execution
  MODULAR_MAX_DEBUG_ASSERT_LEVEL=all -- enables runtime assertions

GPU OOB tests run in subprocesses because the CUDA fault poisons the
context for the rest of the process.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from test_common import graph_error_messages_scenario
from test_common.graph_error_messages_scenario import Scenario

# Locate the scenario script via its module file rather than a hardcoded
# sibling path: it lives in test_common, not next to this test.
_SCENARIO_SCRIPT = Path(graph_error_messages_scenario.__file__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonicalize_region_name(msg: str) -> str:
    """Replace kernel entry point region names with a stable placeholder."""
    return re.sub(
        r'(in kernel entry point named ")[^"]*(")',
        r"\1<region>\2",
        msg,
    )


def _canonicalize_source_traceback(msg: str) -> str:
    """Replace Source Traceback block contents with a stable placeholder.

    Replaces everything between each 'Source Traceback:\\n' header and the
    next blank line ('\\n\\n') or end of string with '<traceback>', making
    assertions independent of file paths, line numbers, and source code.
    """
    return re.sub(
        r"(Source Traceback:\n).+?(\n\n|\Z)",
        r"\1<traceback>\2",
        msg,
        flags=re.DOTALL,
    )


def _canonicalize_types(msg: str) -> str:
    """Replace !mo.tensor<...> type signatures with <type>."""
    return re.sub(r"!mo\.tensor<[^>]+>", "<type>", msg)


def _canonicalize_gpu_error(msg: str) -> str:
    """Canonicalize GPU runtime error messages for stable comparison.

    Splits the message on known section headers, then replaces
    implementation-detail contents within each section while preserving
    headers and the final Note line exactly.
    """
    _SECTION_HEADERS = ("Fusion info:\n", "Source Traceback:\n", "Note: ")

    def _split_sections(text: str) -> list[tuple[str, str]]:
        """Split *text* into (header, body) pairs on known section headers.

        Returns a list where the first entry has header="" (the preamble
        before any known header) and subsequent entries carry the matched
        header string.
        """
        parts: list[tuple[str, str]] = []
        remaining = text
        while remaining:
            earliest_pos = len(remaining)
            earliest_hdr = ""
            for hdr in _SECTION_HEADERS:
                pos = remaining.find(hdr)
                if pos != -1 and pos < earliest_pos:
                    earliest_pos = pos
                    earliest_hdr = hdr
            if not earliest_hdr:
                parts.append(("", remaining))
                break
            if earliest_pos > 0:
                parts.append(("", remaining[:earliest_pos]))
            body_start = earliest_pos + len(earliest_hdr)
            next_pos = len(remaining)
            for hdr in _SECTION_HEADERS:
                pos = remaining.find(hdr, body_start)
                if pos != -1 and pos < next_pos:
                    next_pos = pos
            parts.append((earliest_hdr, remaining[body_start:next_pos]))
            remaining = remaining[next_pos:]
        return parts

    sections = _split_sections(msg)
    out: list[str] = []

    for header, body in sections:
        if header == "":
            body = _canonicalize_region_name(body)
            body = re.sub(
                r'(An error occurred in kernel named ")[^"]*(")',
                r"\1<kernel>\2",
                body,
            )
            body = re.sub(
                r'(An error occurred in kernel named "<kernel>":\n).+',
                r"\1<cuda_error>",
                body,
                flags=re.DOTALL,
            )
            out.append(body)
        elif header == "Fusion info:\n":
            out.append(header + "<fusion_info>")
        elif header == "Source Traceback:\n":
            out.append(header + "<traceback>")
        elif header.startswith("Note: "):
            out.append(header + body)

    return "\n\n".join(out)


def _collect_error_chain(exc: BaseException) -> str:
    """Walk the exception chain and concatenate all messages."""
    parts: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        parts.append(str(current))
        current = current.__cause__
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Compile error tests
# ---------------------------------------------------------------------------


def test_unfused_build_error() -> None:
    """CPU: ops.add on rank-9 tensors triggers compile error (rank > kMaxRank)."""
    shape_9d = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    a_type = TensorType(DType.float32, shape=shape_9d, device=DeviceRef.CPU())
    b_type = TensorType(DType.float32, shape=shape_9d, device=DeviceRef.CPU())

    with Graph("rank9_add", input_types=(a_type, b_type)) as graph:
        a, b = (inp.tensor for inp in graph.inputs)
        result = ops.add(a, b)
        graph.output(result)

    session = InferenceSession()
    with pytest.raises(RuntimeError) as exc_info:
        session.load(graph)

    full_message = _collect_error_chain(exc_info.value)
    canonicalized = _canonicalize_types(
        _canonicalize_source_traceback(full_message)
    )

    expected = (
        "Failed to compile the model. Please file an issue, "
        "all models should be correct by construction and "
        "this error should have been caught during construction.\n"
        "Graph compilation failed:\n"
        "\n"
        "Source Traceback:\n"
        "<traceback>\n"
        "\n"
        "error: 'mo.add' op [MO_TO_MOGG] Found at least one incompatible "
        "input/output type for a JIT-able operation which is not supported. "
        "Operand types: <type>, <type>. "
        "Results Types: <type>\n"
    )
    assert canonicalized == expected


@pytest.mark.skipif(accelerator_count() == 0, reason="Requires GPU")
def test_fused_build_error() -> None:
    """GPU: rank-9 tensors in fused region trigger compile error."""
    gpu = DeviceRef.GPU(0)
    shape_9d = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    data_type = TensorType(DType.float32, shape=shape_9d, device=gpu)
    indices_type = TensorType(DType.int32, shape=shape_9d, device=gpu)
    b_type = TensorType(DType.float32, shape=shape_9d, device=gpu)

    with Graph(
        "rank9_fused", input_types=(data_type, indices_type, b_type)
    ) as graph:
        data, indices, b = (inp.tensor for inp in graph.inputs)
        gathered = ops.gather(data, indices, axis=0)
        summed = ops.add(gathered, b)
        difference = ops.sub(gathered, b)
        product = ops.mul(summed, difference)
        graph.output(product)

    session = InferenceSession(devices=[Accelerator(0)])
    with pytest.raises(RuntimeError) as exc_info:
        session.load(graph)

    full_message = _collect_error_chain(exc_info.value)
    canonicalized = _canonicalize_types(
        _canonicalize_source_traceback(full_message)
    )

    expected = (
        "Failed to compile the model. Please file an issue, "
        "all models should be correct by construction and "
        "this error should have been caught during construction.\n"
        "Graph compilation failed:\n"
        "\n"
        "Source Traceback:\n"
        "<traceback>\n"
        "\n"
        "error: 'mo.gather' op [MO_TO_MOGG] Found at least one incompatible "
        "input/output type for a JIT-able operation which is not supported. "
        "Operand types: <type>, <type>. "
        "Results Types: <type>\n"
    )
    assert canonicalized == expected


# ---------------------------------------------------------------------------
# Runtime assertion tests
# ---------------------------------------------------------------------------


def test_unfused_assert() -> None:
    """CPU: symbolic dimension mismatch across inputs triggers runtime assert."""
    a_type = TensorType(DType.float32, shape=["N", 4], device=DeviceRef.CPU())
    b_type = TensorType(DType.float32, shape=["N", 4], device=DeviceRef.CPU())

    with Graph("symbolic_dim_assert", input_types=(a_type, b_type)) as graph:
        a, b = (inp.tensor for inp in graph.inputs)
        result = ops.add(a, b)
        graph.output(result)

    session = InferenceSession()
    model = session.load(graph)

    a_data = Buffer.from_numpy(np.ones((2, 4), dtype=np.float32))
    b_data = Buffer.from_numpy(np.ones((3, 4), dtype=np.float32))

    with pytest.raises(Exception) as exc_info:
        model.execute(a_data, b_data)

    message = str(exc_info.value)
    canonicalized = _canonicalize_region_name(
        _canonicalize_source_traceback(message)
    )

    expected = (
        'An error occurred in kernel entry point named "<region>":\n'
        "\n"
        "Source Traceback:\n"
        "<traceback>\n"
        "\n"
        "error: symbolic dimension 'N' for input 1 does not match prior"
        " uses of that dimension\n"
    )
    assert canonicalized == expected


@pytest.mark.skipif(accelerator_count() == 0, reason="Requires GPU")
def test_fused_assert() -> None:
    """GPU: symbolic dimension mismatch in fused kernel triggers runtime assert."""
    gpu = DeviceRef.GPU(0)

    data_type = TensorType(DType.float32, shape=["N"], device=gpu)
    indices_type = TensorType(DType.int32, shape=["N"], device=gpu)
    b_type = TensorType(DType.float32, shape=["N"], device=gpu)

    with Graph(
        "symbolic_dim_fused", input_types=(data_type, indices_type, b_type)
    ) as graph:
        data, indices, b = (inp.tensor for inp in graph.inputs)
        gathered = ops.gather(data, indices, axis=0)
        summed = ops.add(gathered, b)
        difference = ops.sub(gathered, b)
        product = ops.mul(summed, difference)
        graph.output(product)

    session = InferenceSession(devices=[Accelerator(0)])
    model = session.load(graph)

    data_buf = Buffer.from_numpy(
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    ).to(Accelerator(0))
    bad_indices = Buffer.from_numpy(np.array([0, 1, 2], dtype=np.int32)).to(
        Accelerator(0)
    )
    b_buf = Buffer.from_numpy(
        np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    ).to(Accelerator(0))

    with pytest.raises(Exception) as exc_info:
        model.execute(data_buf, bad_indices, b_buf)

    message = str(exc_info.value)
    canonicalized = _canonicalize_region_name(
        _canonicalize_source_traceback(message)
    )

    expected = (
        'An error occurred in kernel entry point named "<region>":\n'
        "\n"
        "Source Traceback:\n"
        "<traceback>\n"
        "\n"
        "error: symbolic dimension 'N' for input 1 does not match prior"
        " uses of that dimension\n"
    )
    assert canonicalized == expected


# ---------------------------------------------------------------------------
# GPU OOB gather tests (subprocess isolation)
# ---------------------------------------------------------------------------

_scenario_cache: dict[Scenario, str] = {}


def _run_scenario_in_subprocess(scenario: Scenario) -> str:
    env = os.environ.copy()
    env["_GRAPH_ERROR_SCENARIO"] = scenario.value
    result = subprocess.run(
        [sys.executable, str(_SCENARIO_SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    # GPU kernel asserts may print diagnostics to stdout before the
    # JSON line emitted by the scenario script. Extract the last JSON line.
    for line in reversed(result.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return line
    return result.stdout.strip()


def _warm_all_scenarios() -> None:
    if _scenario_cache:
        return
    with ThreadPoolExecutor(max_workers=len(Scenario)) as pool:
        futures = {
            s: pool.submit(_run_scenario_in_subprocess, s) for s in Scenario
        }
        for s, fut in futures.items():
            _scenario_cache[s] = fut.result()


def _get_scenario_error(scenario: Scenario) -> str:
    _warm_all_scenarios()
    output = _scenario_cache[scenario]
    assert output, f"Subprocess produced no output for {scenario.value}"
    data = json.loads(output)
    error_msg = data.get("error", "")
    assert error_msg, f"No error captured for {scenario.value}"
    return error_msg


@pytest.mark.skipif(accelerator_count() == 0, reason="Requires GPU")
def test_unfused_oob_gather() -> None:
    """GPU: single gather with OOB indices (no fusion)."""
    error_msg = _get_scenario_error(Scenario.UNFUSED_OOB_GATHER)
    canonicalized = _canonicalize_gpu_error(error_msg)

    expected = (
        'An error occurred in kernel entry point named "<region>":\n'
        'An error occurred in kernel named "<kernel>":\n'
        "<cuda_error>\n"
        "\n"
        "Fusion info:\n"
        "<fusion_info>\n"
        "\n"
        "Source Traceback:\n"
        "<traceback>"
    )
    assert canonicalized == expected, (
        f"Mismatch for {Scenario.UNFUSED_OOB_GATHER.value}.\n"
        f"Expected:\n{expected!r}\n\nGot:\n{canonicalized!r}"
    )


@pytest.mark.skipif(accelerator_count() == 0, reason="Requires GPU")
def test_fused_oob_gather() -> None:
    """GPU: gather + add + sub + mul with OOB indices (fused)."""
    error_msg = _get_scenario_error(Scenario.FUSED_OOB_GATHER)
    canonicalized = _canonicalize_gpu_error(error_msg)

    expected = (
        'An error occurred in kernel entry point named "<region>":\n'
        'An error occurred in kernel named "<kernel>":\n'
        "<cuda_error>\n"
        "\n"
        "Fusion info:\n"
        "<fusion_info>\n"
        "\n"
        "Source Traceback:\n"
        "<traceback>\n"
        "\n"
        "Note: 4 distinct Python source locations contributed ops"
        " to this fused kernel."
        " The source traceback for each location is above.\n"
    )
    assert canonicalized == expected, (
        f"Mismatch for {Scenario.FUSED_OOB_GATHER.value}.\n"
        f"Expected:\n{expected!r}\n\nGot:\n{canonicalized!r}"
    )


@pytest.mark.skipif(accelerator_count() == 0, reason="Requires GPU")
def test_fused_oob_gather_duplicate_sources() -> None:
    """GPU: gather + 11x3 elementwise loop (34 ops, 4 unique sources)."""
    error_msg = _get_scenario_error(Scenario.FUSED_OOB_GATHER_DUPLICATE_SOURCES)
    canonicalized = _canonicalize_gpu_error(error_msg)

    expected = (
        'An error occurred in kernel entry point named "<region>":\n'
        'An error occurred in kernel named "<kernel>":\n'
        "<cuda_error>\n"
        "\n"
        "Fusion info:\n"
        "<fusion_info>\n"
        "\n"
        "Source Traceback:\n"
        "<traceback>\n"
        "\n"
        "Note: 4 distinct Python source locations contributed ops"
        " to this fused kernel."
        " The source traceback for each location is above.\n"
    )
    assert canonicalized == expected, (
        f"Mismatch for {Scenario.FUSED_OOB_GATHER_DUPLICATE_SOURCES.value}.\n"
        f"Expected:\n{expected!r}\n\nGot:\n{canonicalized!r}"
    )


@pytest.mark.skipif(accelerator_count() == 0, reason="Requires GPU")
def test_fused_oob_gather_unrolled_sources() -> None:
    """GPU: gather + 11x3 unrolled elementwise (34 ops, 34 unique sources)."""
    error_msg = _get_scenario_error(Scenario.FUSED_OOB_GATHER_UNROLLED_SOURCES)
    canonicalized = _canonicalize_gpu_error(error_msg)

    expected = (
        'An error occurred in kernel entry point named "<region>":\n'
        'An error occurred in kernel named "<kernel>":\n'
        "<cuda_error>\n"
        "\n"
        "Fusion info:\n"
        "<fusion_info>\n"
        "\n"
        "Source Traceback:\n"
        "<traceback>\n"
        "\n"
        "Note: 34 distinct Python source locations contributed ops"
        " to this fused kernel."
        " Only the first 32 source tracebacks are shown above.\n"
    )
    assert canonicalized == expected, (
        f"Mismatch for {Scenario.FUSED_OOB_GATHER_UNROLLED_SOURCES.value}.\n"
        f"Expected:\n{expected!r}\n\nGot:\n{canonicalized!r}"
    )
