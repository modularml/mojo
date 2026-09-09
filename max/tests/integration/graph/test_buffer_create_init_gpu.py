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
"""GPU tests for ``ops.buffer_create`` with an init value.

The device-adaptive checks live in ``buffer_create_init_shared`` and are shared
with the CPU target (``test_buffer_create_init``).
"""

import pytest
from buffer_create_init_shared import (
    make_session,
    run_nonzero_init_value,
    run_persists_across_calls,
)
from max.engine import InferenceSession


@pytest.fixture(scope="module")
def session() -> InferenceSession:
    return make_session()


def test_buffer_create_init_persists_across_calls(
    session: InferenceSession,
) -> None:
    run_persists_across_calls(session)


def test_buffer_create_init_nonzero_init_value(
    session: InferenceSession,
) -> None:
    run_nonzero_init_value(session)
