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

# RUN: %parse-mojo-isolated -verify-diagnostics -split-input-file %s


# expected-error @below {{'main()' does not accept arguments; remove them}}
def main(arg: Int):
    return


# // -----


# expected-error @below {{'main()' does not accept arguments; remove them}}
def main(arg: Int) raises:
    return


# // -----


# expected-error @below {{'main()' does not return a value; remove the return type}}
def main() -> Int:
    return 10


# // -----


# expected-error @below {{'main()' does not accept parameters; remove the square brackets}}
def main[input: Int]():
    return


# // -----


# expected-error @below {{'main' can only be exported as 'main'}}
@export("foo")
def main() abi("Mojo"):
    return


# // -----


# expected-error @below {{only 'main' can be exported as 'main'}}
@export("main")
def fooMain() abi("Mojo"):
    return
