# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# RUN: %mojo %s

from testing import assert_equal


def test_str_none():
    # TODO(#3393): Change to str(None) when MLIR types do not confuse overload resolution.
    # The error we are receiving with str(None) is:
    # cannot bind MLIR type 'None' to trait 'RepresentableCollectionElement'
    assert_equal(str(NoneType()), "None")


def main():
    test_str_none()
