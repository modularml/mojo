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

# RUN: not %mojo %s 2>&1 | FileCheck %s

# Test that `Bool` cannot be used as a collection index.
# This guards against `Bool` accidentally re-gaining `Indexer` conformance,
# e.g. if it ever becomes `SIMD[DType.bool, 1]`.


# CHECK: argument type 'Bool' does not conform to trait 'Indexer'
def main():
    var items = List[Int]()
    _ = items[True]
