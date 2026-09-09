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

# Test --timing-json and --timing-file. The two compose: `--timing-json`
# selects the format, `--timing-file` selects the destination, and either can
# be given on its own.

# A warm cache skips the pass pipeline and reports no passes, so every run
# below needs its own empty MODULAR_CACHE_DIR.

# JSON to a file. Both reports become members of one object, and stderr stays
# clean.
# RUN: rm -rf %t.both %t.both.json && env MODULAR_CACHE_DIR=%t.both %mojo-build --mlir-timing --llvm-timing --timing-json --timing-file=%t.both.json %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK_QUIET --allow-empty
# RUN: FileCheck %s --check-prefix=CHECK_BOTH --input-file=%t.both.json

# One timing option alone gives one member.
# RUN: rm -rf %t.mlir %t.mlir.json && env MODULAR_CACHE_DIR=%t.mlir %mojo-build --mlir-timing --timing-json --timing-file=%t.mlir.json %s -o %t
# RUN: FileCheck %s --check-prefix=CHECK_MLIR_ONLY --input-file=%t.mlir.json

# JSON without a file goes to stderr, so the format does not imply a file.
# RUN: rm -rf %t.stderr && env MODULAR_CACHE_DIR=%t.stderr %mojo-build --mlir-timing --timing-json %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK_MLIR_ONLY

# A file without --timing-json takes the text reports, so the destination does
# not imply a format.
# RUN: rm -rf %t.text %t.text.log && env MODULAR_CACHE_DIR=%t.text %mojo-build --mlir-timing --llvm-timing --timing-file=%t.text.log %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK_QUIET --allow-empty
# RUN: FileCheck %s --check-prefix=CHECK_TEXT --input-file=%t.text.log

# JSON holds an object even if the command asks for no timing.
# RUN: rm -rf %t.none %t.none.json && env MODULAR_CACHE_DIR=%t.none %mojo-build --timing-json --timing-file=%t.none.json %s -o %t
# RUN: FileCheck %s --check-prefix=CHECK_NONE --input-file=%t.none.json

# A file that does not open is an error.
# RUN: not %mojo-build --mlir-timing --timing-file=%t.missing-dir/x.json %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK_BAD_FILE

# CHECK_QUIET-NOT: Execution time report
# CHECK_QUIET-NOT: timing report

# The MLIR member keeps the tree the text report shows; the LLVM member holds
# one named entry per pipeline.
# CHECK_BOTH: "mlir": [
# CHECK_BOTH: "name": "Import Mojo"
# CHECK_BOTH: "llvm": [
# CHECK_BOTH: "pipeline": "host
# CHECK_BOTH: "time.
# CHECK_BOTH: }

# CHECK_MLIR_ONLY: "mlir": [
# CHECK_MLIR_ONLY-NOT: "llvm":

# The text reports keep their titles, which tell the two reports apart.
# CHECK_TEXT: MLIR pass timing (--mlir-timing)
# CHECK_TEXT: Execution time report
# CHECK_TEXT: LLVM pass timing (--llvm-timing): host
# CHECK_TEXT: Pass execution timing report

# CHECK_NONE: {}

# CHECK_BAD_FILE: error: unable to open timing file


def main():
    pass
