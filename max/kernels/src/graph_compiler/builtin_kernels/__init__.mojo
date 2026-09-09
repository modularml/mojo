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

"""Provides the graph compiler's built-in kernel registrations.

Each module binds graph op names (such as `mo.matmul`) to the open-source
kernels in `linalg`, `nn`, `comm`, and related packages using the same
`@extensibility.register` mechanism available to custom ops, so these
registrations double as worked examples of how kernels connect to MAX
graphs.
"""

from .attention import *
from .conv import *
from .distributed import *
from .elementwise import *
from .ep import *
from .gather_scatter import *
from .kernels import *
from .kv_cache import *
from .mtp import *
from .linalg import *
from .quantization import *
from .reductions import *
