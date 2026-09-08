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
# This file is a test input that defines a module within a nested package.


def nested_function():
    return


# A function with a non-inferrable parameter, used to check that an "Included
# from" stack flows through a re-exporting module down to the definition.
def parametric_fn[n: Int]() -> Int:
    return n
