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

"""A package whose modules must import their siblings explicitly.

The re-export below makes `producer` reachable as `no_sibling_leak.producer`
from outside, but a *sibling* module still cannot see it without its own
import - re-exports live in __init__'s scope, not the package scope a
contained file walks up into.
"""

from . import producer
from .producer import reexported_fn
