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
"""Implements CPU algorithm backend utilities including reduction and parallelization."""

from .elementwise import _elementwise_impl_cpu
from .reduction import (
    _reduce_along_inner_dimension,
    _reduce_along_outer_dimension,
    _reduce_generator_cpu,
)
from .parallelize import (
    _get_num_workers,
    parallelize,
    parallelize_over_rows,
    sync_parallelize,
)
from .stencil import _stencil_impl_cpu
