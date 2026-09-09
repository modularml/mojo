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
comptime VALUE = 10


def scope_me():
    print(VALUE)  # prints 10
    comptime VALUE = 20
    # comptime VALUE = 30  # error: invalid redeclaration of VALUE
    comptime if True:
        comptime VALUE = 40
        print(VALUE)  # prints 40
    print(VALUE)  # prints 20


def main():
    scope_me()
