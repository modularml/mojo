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
#
# Makes sure that behaviour of passing call_location() with default inline_level
# of 1 doesn't change, so assert errors like bounds checks retain pointing to
# the source location where the user provided the incorrect index.
#
# ===----------------------------------------------------------------------=== #


from std.reflection import SourceLocation, call_location


# CHECK-LABEL: test_fail
def main():
    print("== test_fail")
    # CHECK: test_debug_assert_location.mojo:28:10: Assert Error: forcing failure
    outer()
    # CHECK-NOT: is never reached
    print("is never reached")


@always_inline
def outer():
    inner(call_location())


@always_inline
def inner(location: SourceLocation):
    debug_assert(False, "forcing failure", location=location)
