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


@always_inline
def nested_callee(a: Int):
    var nested_var = a
    print(nested_var)  # breakpoint


@always_inline("nodebug")
def nodebug_wrapper(b: Int):
    nested_callee(b)


def main():
    nodebug_wrapper(2)
