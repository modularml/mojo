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

"""Experimental APIs for building, sharding, and running ML workloads.

Built on top of :mod:`max.graph` and :mod:`max.driver`, in three layers that
each consume the one below:

- :mod:`~max.experimental.nn` -- the ``Module`` base class plus
  ahead-of-time compilation to a ``CompiledModel``.
- :mod:`~max.experimental.functional` -- a one-function-per-op distributed
  dispatcher (``F.matmul``, ``F.add``, ...).
- :mod:`~max.experimental.sharding` -- placements, the device mesh, the
  action data model, a cost model, and pluggable per-op solvers.

The distributed :class:`~max.experimental.tensor.Tensor` ties them together.

Example:

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor.ones((4, 8))
    y = F.matmul(x, x.T)

.. caution::

    Experimental. The public API is not stable; names and module structure
    may change.
"""

from . import functional, random, tensor, tree_utils
from .tensor import Tensor

__all__ = ["Tensor", "functional", "random", "tensor", "tree_utils"]
