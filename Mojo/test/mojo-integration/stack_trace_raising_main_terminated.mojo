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

# UNSUPPORTED: asan, NVIDIA-GPU, AMD-GPU, system-darwin

# COM: Stack traces are supported on Darwin, but result to different output.
# COM: To avoid having fragile test, mark this test as unsupported on MacOS


def main() raises:
    var x = List[Int]()

    print(x.unsafe_ptr()[unsafe_offset=10000000000])


# RUN: %mojo-build-no-debug-no-assert %s --debug-level full -o %t 2>&1
# RUN: %t 2> %t.log || true
# RUN: cat %t.log | FileCheck --check-prefix=O3-FULL %s

# RUN: %mojo-build-no-debug-no-assert %s --debug-level none -o %t 2>&1
# RUN: %t 2> %t.log || true
# RUN: cat %t.log | FileCheck --check-prefix=O3-NONE %s

# RUN: %mojo-build-no-debug-no-assert %s -O0 --debug-level full -o %t 2>&1
# RUN: %t 2> %t.log || true
# RUN: cat %t.log | FileCheck --check-prefix=O0-FULL %s

# RUN: %mojo-build-no-debug-no-assert %s -O0 --debug-level none -o %t 2>&1
# RUN: %t 2> %t.log || true
# RUN: cat %t.log | FileCheck --check-prefix=O0-NONE %s

# RUN: %mojo-build-no-debug-no-assert %s -O0 --debug-level full -sanitize address -o %t 2>&1
# RUN: %t 2> %t.log || true
# RUN: cat %t.log | FileCheck --check-prefix=O0-FULL-STACK-TRACE-ASAN %s

# RUN: %mojo-build-no-debug-no-assert %s -O0 --debug-level full -sanitize address -o %t 2>&1
# RUN: rm -rf %t-modular-home && mkdir -p %t-modular-home
# RUN: printf '[max-debug]\nstack-trace-on-crash = false\n' > %t-modular-home/modular.cfg
# RUN: env MODULAR_HOME=%t-modular-home %t 2> %t.log || true
# RUN: cat %t.log
# RUN: cat %t.log | FileCheck --check-prefix=O0-FULL-NO-STACK-TRACE-ASAN %s

# RUN: mojo run %s 2> %t.log || true
# RUN: cat %t.log | FileCheck --check-prefix=O3-JIT-HELP-MESSAGE %s

# O3-FULL: #{{.*}} stack_trace_raising_main_terminated{{.*}}::main() {{.*}}/stack_trace_raising_main_terminated.mojo:{{.*}}:{{.*}}
# O3-FULL: #{{.*}} std::builtin::_startup::__wrap_and_execute_raising_main{{.*}}raises{{.*}}main() {{.*}}_startup.mojo:{{.*}}:{{.*}}

# O3-NONE: #{{.*}} llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) {{.*}}Signals
# O3-NONE: #{{.*}} llvm::sys::RunSignalHandlers() {{.*}}Signals.cpp
# O3-NONE: #{{.*}} main

# O0-FULL: #{{.*}} llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) {{.*}}Signals
# O0-FULL: #{{.*}} llvm::sys::RunSignalHandlers() {{.*}}Signals.cpp
# O0-FULL: #{{.*}} stack_trace_raising_main_terminated{{.*}}::main() {{.*}}/stack_trace_raising_main_terminated.mojo:{{.*}}:{{.*}}
# O0-FULL: #{{.*}} std::builtin::_startup::__wrap_and_execute_raising_main{{.*}}raises{{.*}}main() {{.*}}_startup.mojo:{{.*}}:{{.*}}

# O0-NONE: #{{.*}} llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) {{.*}}Signals
# O0-NONE: #{{.*}} llvm::sys::RunSignalHandlers() {{.*}}Signals.cpp
# O0-NONE: #{{.*}} stack_trace_raising_main_terminated{{.*}}::main() stack_trace_raising_main_terminated.mojo:{{.*}}:{{.*}}
# O0-NONE: #{{.*}} std::builtin::_startup::__wrap_and_execute_raising_main{{.*}}raises{{.*}}main()
# O0-NONE: #{{.*}} main

# O0-FULL-STACK-TRACE-ASAN: PrintStackTrace
# O0-FULL-STACK-TRACE-ASAN: AddressSanitizer: SEGV on unknown address

# O0-FULL-NO-STACK-TRACE-ASAN-NOT: PrintStackTrace
# O0-FULL-NO-STACK-TRACE-ASAN: AddressSanitizer: SEGV on unknown address

# O3-JIT-HELP-MESSAGE: To get a symbolicated stack trace, compile your program using `mojo build` with debug info enabled (e.g., `-debug-level=line-tables`) and execute it separately.
