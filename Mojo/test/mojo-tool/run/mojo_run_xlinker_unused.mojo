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

# `mojo run` has no native linker (the program is JIT'd in-process). Flags
# that are not recognized as a library reference, or `-l<name>` references
# that don't resolve, are reported as non-fatal warnings and dropped.

# RUN: mojo run -Xlinker some-bogus-flag %s 2>&1 \
# RUN:   | FileCheck %s --check-prefixes WARN,CHECK
# RUN: mojo run -Xlinker some-bogus-flag --disable-warnings %s 2>&1 \
# RUN:   | FileCheck %s --check-prefixes NO-WARN,CHECK

# RUN: mojo run -Xlinker -ldoes-not-exist %s 2>&1 \
# RUN:   | FileCheck %s --check-prefixes MISSING,CHECK

# A path that exists but does not look like a shared library (e.g. this
# source file itself) should be rejected, not silently dlopen'd.
# RUN: mojo run -Xlinker %s %s 2>&1 \
# RUN:   | FileCheck %s --check-prefixes NOT-LIBRARY,CHECK

# A path that looks like a shared library but does not exist should be
# called out as missing rather than as an unrecognized flag.
# RUN: mojo run -Xlinker /tmp/does-not-exist.so %s 2>&1 \
# RUN:   | FileCheck %s --check-prefixes MISSING-LIB,CHECK

# WARN: warning: -Xlinker argument has no effect on `mojo run`: 'some-bogus-flag'
# MISSING: warning: could not locate shared library for '-ldoes-not-exist'
# NOT-LIBRARY: warning: -Xlinker argument has no effect on `mojo run`:
# NOT-LIBRARY-SAME: mojo_run_xlinker_unused.mojo'
# MISSING-LIB: warning: shared library does not exist for `-Xlinker` argument: '/tmp/does-not-exist.so'
# NO-WARN-NOT: warning
# CHECK: hello, world


def main() raises:
    print("hello, world")
