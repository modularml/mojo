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

# RUN: mkdir %t
# RUN: %mojo-build %s -o %t/output.o --emit object

# COM: Check that `file` recognizes it as an object file
# RUN: file %t/output.o | FileCheck %s

# COM: Full output:
# COM:   - macOS: "output.o: Mach-O 64-bit object arm64"
# COM:   - Linux: "output.o: ELF 64-bit LSB relocatable, ARM aarch64, version 1 (SYSV), not stripped"
# CHECK: output.o: {{(Mach-O 64-bit object|ELF 64-bit LSB relocatable)}}


def main():
    pass
