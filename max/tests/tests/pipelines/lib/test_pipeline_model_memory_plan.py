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
"""Pins the PipelineModel memory-plan contract.

The effective sequence-length bound has one home, the memory plan: a
pipeline model cannot be built without one, and ``max_seq_len`` is a
read-only view of the plan rather than a stored copy with a derived
fallback.
"""

import inspect
from types import SimpleNamespace

from max.pipelines.lib import MemoryPlan, PipelineModel


def test_memory_plan_is_a_required_keyword() -> None:
    param = inspect.signature(PipelineModel.__init__).parameters["memory_plan"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is inspect.Parameter.empty


def test_max_seq_len_is_a_read_only_view_of_the_plan() -> None:
    descriptor = inspect.getattr_static(PipelineModel, "max_seq_len")
    assert isinstance(descriptor, property)
    assert descriptor.fset is None

    fget = descriptor.fget
    assert fget is not None
    plan = MemoryPlan(
        planned_max_batch_size=1, footprint=0, planned_max_length=77
    )
    assert fget(SimpleNamespace(memory_plan=plan)) == 77
