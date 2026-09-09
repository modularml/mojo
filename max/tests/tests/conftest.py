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

import os
from collections.abc import Generator

import pytest
from max import mlir
from max._mlir_context import default_mlir_context
from max.driver import set_virtual_cpu_target

# Set at collection time, before any test imports max._interpreter_ops.
_test_cpu_target = os.environ.get("MODULAR_TEST_VIRTUAL_CPU_TARGET")
if _test_cpu_target:
    set_virtual_cpu_target(_test_cpu_target)


@pytest.fixture(scope="function")
def mlir_context() -> Generator[mlir.Context]:
    """Set up the MLIR context by registering and loading Modular dialects."""
    ctx = default_mlir_context()
    with mlir.Location.unknown():
        yield ctx
