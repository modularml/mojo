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


def foo() raises:
    raise Error("gotcha!")


@no_inline
def nested_func() raises:
    foo1()


@no_inline
def foo1() raises:
    foo2()


@no_inline
def foo2() raises:
    raise Error("nested gotcha!")


def main():
    try:
        foo()
    except e:
        print("stack trace of", e)
        var stack_trace = e.get_stack_trace()
        if stack_trace:
            print(stack_trace.value())
        else:
            print(
                "stack trace was not collected. Enable stack trace collection"
                " with environment variable"
                " `MODULAR_DEBUG=stack-trace-on-error`"
            )

    try:
        nested_func()
    except e:
        print("stack trace of", e)
        var stack_trace = e.get_stack_trace()
        if stack_trace:
            print(stack_trace.value())
        else:
            print(
                "stack trace was not collected. Enable stack trace collection"
                " with environment variable"
                " `MODULAR_DEBUG=stack-trace-on-error`"
            )


# RUN: %mojo-build-no-debug-no-assert %s --debug-level full -o %t 2>&1
# RUN: MODULAR_DEBUG=stack-trace-on-error %t > %t.log
# RUN: cat %t.log | FileCheck --check-prefix=O3-FULL %s
# RUN: %t > %t.log
# RUN: cat %t.log | FileCheck --check-prefix=O3-FULL-NO-STACK %s

# RUN: %mojo-build-no-debug-no-assert %s --debug-level none -o %t 2>&1
# RUN: MODULAR_DEBUG=stack-trace-on-error %t > %t.log
# RUN: cat %t.log | FileCheck --check-prefix=O3-NONE %s

# RUN: %mojo-build-no-debug-no-assert %s -O0 --debug-level full -o %t 2>&1
# RUN: MODULAR_DEBUG=stack-trace-on-error %t > %t.log
# RUN: cat %t.log | FileCheck --check-prefix=O0-FULL %s

# RUN: %mojo-build-no-debug-no-assert %s -O0 --debug-level none -o %t 2>&1
# RUN: MODULAR_DEBUG=stack-trace-on-error %t > %t.log
# RUN: cat %t.log | FileCheck --check-prefix=O0-NONE %s

# O3-FULL-LABEL: stack trace of gotcha!
# O3-FULL:      #{{.*}} KGEN_CompilerRT_GetStackTrace
# O3-FULL-NEXT: #{{.*}} std::builtin::error::StackTrace::collect_if_enabled(::SIMD[DType.int, 1])
# O3-FULL:      #{{.*}} stack_trace_exception::main() {{.*}}/stack_trace_exception.mojo:
# O3-FULL-NEXT: #{{.*}} std::builtin::_startup::__wrap_and_execute_main[def() thin -> None]
# O3-FULL-NEXT: #{{.*}} main Mojo/stdlib/std/builtin/_startup.mojo:

# O3-FULL-LABEL: stack trace of nested gotcha!
# O3-FULL:      #{{.*}} KGEN_CompilerRT_GetStackTrace
# O3-FULL-NEXT: #{{.*}} std::builtin::error::StackTrace::collect_if_enabled(::SIMD[DType.int, 1])
# O3-FULL-NEXT: #{{.*}} stack_trace_exception::foo2()_REMOVED_ARG
# O3-FULL-NEXT: #{{.*}} stack_trace_exception::main() {{.*}}/stack_trace_exception.mojo:
# O3-FULL-NEXT: #{{.*}} std::builtin::_startup::__wrap_and_execute_main
# O3-FULL-NEXT: #{{.*}} main

# O3-FULL-NO-STACK-LABEL: stack trace of gotcha!
# O3-FULL-NO-STACK: stack trace was not collected. Enable stack trace collection with environment variable `MODULAR_DEBUG=stack-trace-on-error`

# O3-FULL-NO-STACK-LABEL: stack trace of nested gotcha!
# O3-FULL-NO-STACK: stack trace was not collected. Enable stack trace collection with environment variable `MODULAR_DEBUG=stack-trace-on-error`

# O3-NONE-LABEL: stack trace of gotcha!
# O3-NONE: #{{.*}} KGEN_CompilerRT_GetStackTrace
# O3-NONE: #{{.*}} main

# O3-NONE-LABEL: stack trace of nested gotcha!
# O3-NONE:      #{{.*}} KGEN_CompilerRT_GetStackTrace
# O3-NONE:      #{{.*}} stack_trace_exception::foo2()_REMOVED_ARG stack_trace_exception.mojo:{{.*}}:{{.*}}
# O3-NONE-NEXT: #{{.*}} stack_trace_exception::foo1() stack_trace_exception.mojo:{{.*}}:{{.*}}
# O3-NONE-NEXT: #{{.*}} main


# O0-FULL-LABEL: stack trace of gotcha!
# O0-FULL:      #{{.*}} KGEN_CompilerRT_GetStackTrace
# O0-FULL-NEXT: #{{.*}} std::builtin::error::StackTrace::collect_if_enabled(::SIMD[DType.int, 1]) Mojo/stdlib/std/builtin/error.mojo:{{.*}}:{{.*}}
# O0-FULL-NEXT: #{{.*}} stack_trace_exception::foo() {{.*}}/stack_trace_exception.mojo:{{.*}}:{{.*}}
# O0-FULL-NEXT: #{{.*}} stack_trace_exception::main() {{.*}}/stack_trace_exception.mojo:{{.*}}:{{.*}}
# O0-FULL-NEXT: #{{.*}} std::builtin::_startup::__wrap_and_execute_main[def() thin -> None]
# O0-FULL-NEXT: #{{.*}} main

# O0-FULL-LABEL: stack trace of nested gotcha!
# O0-FULL:       #{{.*}} KGEN_CompilerRT_GetStackTrace
# O0-FULL-NEXT: #{{.*}} std::builtin::error::StackTrace::collect_if_enabled(::SIMD[DType.int, 1]) Mojo/stdlib/std/builtin/error.mojo:{{.*}}:{{.*}}
# O0-FULL-NEXT:  #{{.*}} stack_trace_exception::foo2() {{.*}}/stack_trace_exception.mojo:{{.*}}:{{.*}}
# O0-FULL-NEXT:  #{{.*}} stack_trace_exception::foo1() {{.*}}/stack_trace_exception.mojo:{{.*}}:{{.*}}
# O0-FULL-NEXT:  #{{.*}} stack_trace_exception::nested_func() {{.*}}/stack_trace_exception.mojo:{{.*}}:{{.*}}
# O0-FULL-NEXT:  #{{.*}} stack_trace_exception::main() {{.*}}/stack_trace_exception.mojo:{{.*}}:{{.*}}
# O0-FULL-NEXT:  #{{.*}} std::builtin::_startup::__wrap_and_execute_main[def() thin -> None]
# O0-FULL-NEXT: #{{.*}} main

# O0-NONE-LABEL: stack trace of gotcha!
# O0-NONE:      #{{.*}} KGEN_CompilerRT_GetStackTrace
# O0-NONE:      #{{.*}} stack_trace_exception::foo() stack_trace_exception.mojo:{{.*}}:{{.*}}
# O0-NONE-NEXT: #{{.*}} stack_trace_exception::main() stack_trace_exception.mojo:{{.*}}:{{.*}}
# O0-NONE-NEXT: #{{.*}} std::builtin::_startup::__wrap_and_execute_main
# O0-NONE-NEXT: #{{.*}} main

# O0-NONE-LABEL: stack trace of nested gotcha!
# O0-NONE:       #{{.*}} KGEN_CompilerRT_GetStackTrace
# O0-NONE:       #{{.*}} stack_trace_exception::foo2() stack_trace_exception.mojo:{{.*}}:{{.*}}
# O0-NONE-NEXT:  #{{.*}} stack_trace_exception::foo1() stack_trace_exception.mojo:{{.*}}:{{.*}}
# O0-NONE-NEXT:  #{{.*}} stack_trace_exception::nested_func() {{.*}}stack_trace_exception.mojo:{{.*}}:{{.*}}
# O0-NONE-NEXT:  #{{.*}} stack_trace_exception::main() stack_trace_exception.mojo:{{.*}}:{{.*}}
# O0-NONE-NEXT:  #{{.*}} std::builtin::_startup::__wrap_and_execute_main
# O0-NONE-NEXT:  #{{.*}} main
