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
# Checks the message and source location reported by a `debug_assert` with no
# message arguments. The location must stay at the caller.
#
# ===----------------------------------------------------------------------=== #


from std.reflection import SourceLocation, call_location


# CHECK-LABEL: test_no_message
def main():
    print("== test_no_message")
    # CHECK: test_debug_assert_no_message.mojo:27:10: Assert Error: assertion failed
    outer()
    # CHECK-NOT: is never reached
    print("is never reached")


@always_inline
def outer():
    inner(call_location())


@always_inline
def inner(location: SourceLocation):
    debug_assert(False, location=location)
