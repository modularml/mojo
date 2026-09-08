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

# REQUIRES: x86_64-linux
# COM: This check only makes sense for generating an ELF object file.
# RUN: %mojo-build %s --mcmodel=medium --large-data-threshold=2 -o %t
# RUN: llvm-objdump %t -t | FileCheck %s
# RUN: %mojo-build %s --emit=asm --relocation-model=pic --mcmodel=medium --large-data-threshold=2 -o - | FileCheck %s --check-prefix=CHECK-PIC
# RUN: %mojo-build %s --emit=asm --relocation-model=static --mcmodel=medium --large-data-threshold=2 -o - | FileCheck %s --check-prefix=CHECK-STATIC


# COM: check that string constant is in .lrodata section
# (for any data size that's larger than large-data-threshold)
# CHECK: .lrodata
# CHECK-PIC: GOTOFF
# CHECK-STATIC-NOT: GOTOFF
#
def main():
    print("hello world.")
