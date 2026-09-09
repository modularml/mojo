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
"""Mock standard library package for testing stability tracking in mojo doc output."""


@stable
struct StableStruct:
    """A stable struct."""

    pass


struct UnstableStruct:
    """An unstable struct."""

    pass


@stable(since="1.0")
def stable_fn_with_version():
    """A stable function with a version string."""
    pass
