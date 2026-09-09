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

# RUN: %mojo %s --debug-level full 2>&1 | FileCheck %s

from std.collections.string import StaticString


def try_it(c0: Bool, c1: Bool) raises -> StaticString:
    try:
        try:
            print("try")
            return "dead code"
        finally:
            print("finally")
            if c0:
                return "true!"
            return "false!"
    finally:
        print("finally again!")
        if c1:
            return "interrupt!"


struct MyCtxtMgr:
    var handle: Bool

    @implicit
    def __init__(out self, handle: Bool = True):
        self.handle = handle

    def __enter__(self):
        pass

    def __exit__(self):
        print("exit!")

    def __exit__(self, err: Error) -> Bool:
        print("exit error!")
        return self.handle


def with_no_throw() -> Int:
    with MyCtxtMgr():
        return 1


def with_it() raises -> Int:
    # CHECK-NOT: warning: 'except' logic is unreachable, try doesn't raise an exception
    with MyCtxtMgr():
        return 2


def with_it_err(handle: Bool) raises -> Int:
    with MyCtxtMgr(handle):
        raise Error()
    return 3


@fieldwise_init
struct MemoryType(ImplicitlyCopyable):
    def __deinit__(deinit self):
        print("delete")


def chris_origin_example(a: Bool, b: Bool):
    print("start")
    var x: MemoryType
    try:
        try:
            if a:
                x = MemoryType()
                raise Error()
        finally:
            if b:
                print("early")
                return
    except:
        _ = x^  # Keep alive for the test
    print("normal")


def raise_fn() raises:
    raise Error("in raise_fn")


def raise_cond() -> Bool:
    return False


def raising_finally():
    try:
        try:
            if raise_cond():
                print("try inner")
                raise_fn()
                return
            else:
                return
        except _:
            print("except inner")
            return
        finally:
            print("finally inner")
            raise_fn()
    except _:
        print("except outer")
        return


@fieldwise_init
struct MoveMe(Copyable):
    pass


def moved(var x: MoveMe):
    print("moved called")
    pass


# We should not report `use of uninitialized value 'x'`
def raising_finally_no_use_of_uninit():
    var x = MoveMe()

    try:
        try:
            raise_fn()

            moved(x^)
            return
        except _:
            moved(x^)
            return
        finally:
            raise_fn()
            return
    except _:
        return


def main() raises:
    # CHECK-LABEL: == try-finally
    print("== try-finally")
    # CHECK-NEXT: try
    # CHECK-NEXT: finally
    # CHECK-NEXT: finally again!
    # CHECK-NEXT: true!
    print(try_it(True, False))
    # CHECK-NEXT: try
    # CHECK-NEXT: finally
    # CHECK-NEXT: finally again!
    # CHECK-NEXT: false!
    print(try_it(False, False))
    # CHECK-NEXT: try
    # CHECK-NEXT: finally
    # CHECK-NEXT: finally again!
    # CHECK-NEXT: interrupt!
    print(try_it(True, True))
    # CHECK-NEXT: try
    # CHECK-NEXT: finally
    # CHECK-NEXT: finally again!
    # CHECK-NEXT: interrupt!
    print(try_it(False, True))

    # CHECK-NEXT: exit!
    # CHECK-NEXT: 1
    print(with_no_throw())
    # CHECK-NEXT: exit!
    # CHECK-NEXT: 2
    print(with_it())
    # CHECK-NEXT: exit error!
    # CHECK-NEXT: 3
    print(with_it_err(True))
    try:
        # CHECK-NEXT: exit error!
        print(with_it_err(False))
    except:
        # CHECK-NEXT: an error was raised
        print("an error was raised")

    # CHECK-NEXT: start
    # CHECK-NEXT: delete
    # CHECK-NEXT: normal
    chris_origin_example(True, False)
    # CHECK-NEXT: start
    # CHECK-NEXT: early
    chris_origin_example(False, True)
    # CHECK-NEXT: start
    # CHECK-NEXT: delete
    # CHECK-NEXT: early
    chris_origin_example(True, True)

    # CHECK-NEXT: finally inner
    # CHECK-NEXT: except outer
    raising_finally()

    # Should only call `def moved` once.
    # CHECK-NEXT: moved called
    # CHECK-NOT: moved called
    raising_finally_no_use_of_uninit()
