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
"""Named by a multi-component relative wildcard in the package __init__."""


# Re-exported by `from .deep.inner import *` in the package __init__, but the
# walk resolves wildcards by leaf name only, so this is NOT found (limitation).
def multicomp_wildcard_symbol():
    pass
