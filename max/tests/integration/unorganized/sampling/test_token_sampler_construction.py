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
"""Graph-construction tests for ``token_sampler`` (no GPU required).

Verifies the bitmask input is wired into the sampler graph iff the caller
asks for it via ``needs_bitmask_input``, independently of
``sampling_config.enable_structured_output``. The override path is what
keeps tool-call grammars working when ``--enable-structured-output`` is
off.
"""

from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.sampling import SamplingConfig, token_sampler
from max.pipelines.sampling.sampling import _sampling_input_types


def _input_keys(
    *, enable_structured_output: bool, needs_bitmask_input: bool
) -> set[str]:
    sampling_config = SamplingConfig(
        enable_structured_output=enable_structured_output,
        in_dtype=DType.float32,
        out_dtype=DType.float32,
    )
    inputs = _sampling_input_types(
        sampling_config,
        return_logits=False,
        device=DeviceRef.CPU(),
        needs_bitmask_input=needs_bitmask_input,
    )
    return set(inputs.keys())


def test_bitmask_input_present_when_flag_on() -> None:
    """Baseline: flag on → bitmask wired in."""
    assert "bitmask" in _input_keys(
        enable_structured_output=True, needs_bitmask_input=True
    )


def test_bitmask_input_absent_when_flag_off_and_no_override() -> None:
    """Baseline: flag off, no override → no bitmask."""
    assert "bitmask" not in _input_keys(
        enable_structured_output=False, needs_bitmask_input=False
    )


def test_bitmask_input_present_when_override_forces_it_with_flag_off() -> None:
    """Regression guard: bitmask wired in when override=True even if the
    sampling-config flag is off.

    This is the tool-call-grammar-without-``--enable-structured-output``
    path. Without this gating the bitmask sampler graph would have no
    ``bitmask`` input and would crash with an arity mismatch at runtime.
    """
    assert "bitmask" in _input_keys(
        enable_structured_output=False, needs_bitmask_input=True
    )


def test_bitmask_input_absent_when_override_suppresses_with_flag_on() -> None:
    """Regression guard: bitmask not wired in when override=False even if
    the sampling-config flag is on. Used for the bitmask-free fallback
    sampler in the pipeline init.
    """
    assert "bitmask" not in _input_keys(
        enable_structured_output=True, needs_bitmask_input=False
    )


def test_token_sampler_default_override_falls_back_to_flag() -> None:
    """``token_sampler(needs_bitmask_input=None)`` preserves backwards-compat
    by falling back to ``sampling_config.enable_structured_output``.

    Construct the graph twice — once with the flag on and once off — and
    verify the input count differs by exactly one (the bitmask input).
    """
    cfg_on = SamplingConfig(
        enable_structured_output=True,
        in_dtype=DType.float32,
        out_dtype=DType.float32,
    )
    cfg_off = SamplingConfig(
        enable_structured_output=False,
        in_dtype=DType.float32,
        out_dtype=DType.float32,
    )
    graph_on = token_sampler(cfg_on, device=DeviceRef.CPU())
    graph_off = token_sampler(cfg_off, device=DeviceRef.CPU())
    assert len(graph_on.inputs) == len(graph_off.inputs) + 1
