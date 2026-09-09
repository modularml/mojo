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
# RUN: kgen --emit=llvm-opt %s | FileCheck %s


# CHECK: ; Function Attrs: {{.*}}memory(argmem: readwrite)
# CHECK-LABEL: @mayalias(
@export
def mayalias(
    a: Pointer[Float32, ImmUnsafeAnyOrigin],
    b: Pointer[Float32, MutAnyOrigin],
) abi("Mojo") -> Float32:
    # CHECK: store
    b[] += a[] * b[]
    # CHECK-NEXT: load
    # CHECK-NEXT: fmul
    return a[] * b[]


# CHECK-LABEL: @noalias(
@export
def noalias(
    a0: Pointer[Float32, ImmUnsafeAnyOrigin],
    b: Pointer[Float32, MutAnyOrigin],
) abi("Mojo") -> Float32:
    var a = a0.unsafe_as_noalias()

    # CHECK: store
    b[] += a[] * b[]
    # CHECK-NEXT: fmul
    return a[] * b[]


# MOCO-914: potentially mutable references are non-aliasing.
# CHECK-LABEL: @any_life(
# CHECK-SAME: ptr noalias nofree noundef nonnull readnone captures(none) %0,
# CHECK-SAME: ptr noalias nofree noundef nonnull readnone captures(none) %1)
@export
def any_life(ref[MutAnyOrigin] r: Int, mut x: Int) abi("Mojo"):
    pass


# CHECK-LABEL: @imm_life(
# CHECK-SAME: ptr nofree noundef nonnull readnone captures(none) %0,
# CHECK-SAME: ptr noalias nofree noundef nonnull readnone captures(none) %1)
@export
def imm_life(ref[ImmUnsafeAnyOrigin] r: Int, mut x: Int) abi("Mojo"):
    pass
