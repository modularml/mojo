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

# Test that the default Writable implementation produces a clear error message
# when a struct has a field with an MLIR type that does not implement Writable.
# Regression test for MSTDL-2340: previously this would crash the compiler with
# "'get_type_name' requires a concrete type".


@fieldwise_init
struct HasMLIRField(Writable):
    var x: Int
    var _internal: __mlir_type.index


# CHECK: constraint failed: Could not derive Writable for HasMLIRField - member field `_internal` does not implement Writable
def main():
    var s = HasMLIRField(x=1, _internal=__mlir_attr[`0 : index`])
    print(s)
