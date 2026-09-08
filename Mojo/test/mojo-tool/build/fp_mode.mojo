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

# Test that `--fp-mode` controls FP contraction on the multiply-add in the
# emitted (unoptimized) LLVM IR: `contract=fast` (the default) keeps the
# `contract` flag so `a*b + c` can fuse into an FMA, while `contract=off`
# strips it.

# RUN: mkdir -p %t

# RUN: %mojo-build %s --fp-mode=contract=fast -o %t/contract.ll --emit llvm
# RUN: FileCheck %s --check-prefix=CONTRACT --input-file=%t/contract.ll

# RUN: %mojo-build %s --fp-mode=contract=off -o %t/precise.ll --emit llvm
# RUN: FileCheck %s --check-prefix=PRECISE --input-file=%t/precise.ll

# The default (no flag) keeps contraction on.
# RUN: %mojo-build %s -o %t/default.ll --emit llvm
# RUN: FileCheck %s --check-prefix=CONTRACT --input-file=%t/default.ll

# A list is processed left-to-right, so the last item wins: `contract=off`
# here, whether given as one comma-separated value or repeated flags.
# RUN: %mojo-build %s --fp-mode=contract=fast,contract=off -o %t/list.ll --emit llvm
# RUN: FileCheck %s --check-prefix=PRECISE --input-file=%t/list.ll

# RUN: %mojo-build %s --fp-mode=contract=fast --fp-mode=contract=off -o %t/rep.ll --emit llvm
# RUN: FileCheck %s --check-prefix=PRECISE --input-file=%t/rep.ll

# An unknown feature is rejected with a diagnostic.
# RUN: not %mojo-build %s --fp-mode=bogus=fast -o %t/bad.ll --emit llvm 2>&1 \
# RUN:   | FileCheck %s --check-prefix=INVALID
# INVALID: invalid fp-mode 'bogus=fast', the only supported feature is 'contract'

# An invalid value is rejected with a diagnostic.
# RUN: not %mojo-build %s --fp-mode=contract=maybe -o %t/bad.ll --emit llvm 2>&1 \
# RUN:   | FileCheck %s --check-prefix=BADVALUE
# BADVALUE: invalid fp-mode 'contract=maybe', expected 'contract=fast' or 'contract=off'

# `contract=fast` keeps the multiply and add fused-able (both carry `contract`).
# CONTRACT: fmul contract float
# CONTRACT: fadd contract float

# `contract=off` strips contraction from both, so no FMA can form.
# PRECISE: fmul float
# PRECISE: fadd float


@no_inline
def madd(a: Float32, b: Float32, c: Float32) -> Float32:
    return a * b + c


def main():
    print(madd(1.0, 2.0, 3.0))
