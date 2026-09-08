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

# UNSUPPORTED: system-darwin, NVIDIA-GPU, AMD-GPU

# COM: Stack traces are supported on Darwin, but result in different output.
# COM: Stack traces are disabled on GPU.
# COM: To avoid having fragile tests, mark this test as unsupported on these platforms.


@no_inline
def nested_func() raises:
    foo1()


@no_inline
def foo1() raises:
    foo2()


@no_inline
def foo2() raises:
    raise Error("nested gotcha!")


def main() raises:
    nested_func()


# RUN: %mojo-build-no-debug-no-assert %s --debug-level full -o %t
# RUN: MODULAR_DEBUG=stack-trace-on-error %t 2> %t.log || true
# RUN: cat %t.log | FileCheck --check-prefix=O3-FULL %s

# RUN: %mojo-build-no-debug-no-assert %s --debug-level full -o %t
# RUN: %t 2> %t.log || true
# RUN: cat %t.log | FileCheck --check-prefix=O3-FULL-NO-STACK %s

# O3-FULL-NO-STACK: stack trace was not collected. Enable stack trace collection with environment variable `MODULAR_DEBUG=stack-trace-on-error`
# O3-FULL-NO-STACK: Unhandled exception caught during execution: nested gotcha!

# O3-FULL:      #{{.*}} KGEN_CompilerRT_GetStackTrace
# O3-FULL-NEXT: #{{.*}} std::builtin::error::StackTrace::collect_if_enabled(::SIMD[DType.int, 1])
# O3-FULL-NEXT: #{{.*}} stack_trace_exception_no_handling::foo2()_REMOVED_ARG {{.*}}/stack_trace_exception_no_handling.mojo:{{.*}}:{{.*}}
# O3-FULL-NEXT: #{{.*}} std::builtin::_startup::__wrap_and_execute_raising_main
# O3-FULL-NEXT: #{{.*}} main Mojo/stdlib/std/builtin/_startup.mojo:{{.*}}:{{.*}}
# O3-FULL: Unhandled exception caught during execution: nested gotcha!
