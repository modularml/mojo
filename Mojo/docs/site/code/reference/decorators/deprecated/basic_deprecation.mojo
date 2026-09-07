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


# Mark function `a` as deprecated with a custom message
@deprecated("Ignore. Deprecation test. Sunsetting a")
def a():
    pass


# Mark function `b` as deprecated with alternative
@deprecated(use=c)
def b():
    pass


# `c` is `b`'s recommended replacement after deprecation
def c():
    pass


def main():
    a()  # custom warning
    b()  # warning with recommended replacement
    c()  # no warning

    # Demonstrate that only warnings are issued
    print("This is a functioning app")
