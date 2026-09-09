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
"""Direct child module whose leaf name (`inner`) collides with the final
component of the package __init__'s `from .deep.inner import *`. Not re-exported
by any __init__, so its symbol is not public API and must get no suggestion. It
exists to pin the leaf-collision bug: resolving that multi-component wildcard by
leaf name alone would false-match this unrelated direct child."""


def collision_symbol():
    pass
