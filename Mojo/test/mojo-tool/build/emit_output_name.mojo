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
#
# Test that when using `--emit`, `mojo build` creates the output files in the
# current working directory, using nice names that are based on the input file
# name.
#
# ===----------------------------------------------------------------------=== #

# RUN: rm %S/emit_output_name.ll || true
# RUN: %mojo-build --emit llvm %s
# RUN: test -e %S/emit_output_name.ll

# RUN: rm %S/emit_output_name.ll || true
# RUN: %mojo-build %s --emit llvm
# RUN: test -e %S/emit_output_name.ll

# RUN: rm %S/emit_output_name.s || true
# RUN: %mojo-build --emit asm %s
# RUN: test -e %S/emit_output_name.s

# RUN: rm %S/emit_output_name.s || true
# RUN: %mojo-build %s --emit asm
# RUN: test -e %S/emit_output_name.s


def main():
    pass
