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

# RUN: %parse-mojo-isolated %s --mlir-print-debuginfo -o %t.mlir
# RUN: kgen-opt %t.mlir -lower-semantic-cf -check-lifetimes -verify-parameters -verify-diagnostics | FileCheck %s
# RUN: %parse-mojo-isolated %s --mlir-print-debuginfo --debug-level full -o /dev/null

# Control flow related CheckLifetimes tests.


def use(err: Error):
    pass


def use(str: String):
    pass


def use(a: MemExample):
    pass


def use_mut(mut a: MemExample):
    pass


# CHECK-LABEL: lit.struct.decl @MemExample
struct MemExample(ImplicitlyCopyable):
    var x: Int

    def __init__(out self):
        self.x = 42
        pass

    def noop(self):
        pass

    def __init__(out self, *, deinit move: Self):
        self.x = move.x

    def __init__(out self, *, copy: Self):
        self.x = copy.x

    def __bool__(self) -> Bool:
        return True

    def unsafe_ptr(self) -> UnsafePointer[Int, AnyOrigin[mut=True]]:
        return UnsafePointer[Int, AnyOrigin[mut=True]].unsafe_dangling()

    def __deinit__(deinit self):
        pass


# CHECK-LABEL: lit.fn @"if_examples
def if_examples(cond: __mlir_type.`!kgen.scalar<bool>`):
    # CHECK: %_a = lit.var.decl
    var _a: MemExample

    # CHECK-NEXT: %_b = lit.var.decl
    # CHECK-NEXT: lifetime.start %_b
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%_b)
    # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%_b)
    # CHECK-NEXT: lifetime.end %_b
    var _b = MemExample()

    # CHECK: hlcf.elif %cond {
    if cond:
        # CHECK-NEXT: lifetime.start %_a
        # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%_a)
        # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%_a)
        # CHECK-NEXT: lifetime.end %_a
        _a = MemExample()
    # CHECK-NEXT: hlcf.yield
    # CHECK-NEXT: } else {
    else:
        # CHECK-NEXT: lifetime.start %_b
        # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%_b)
        # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%_b)
        # CHECK-NEXT: lifetime.end %_b
        _b = MemExample()
    # CHECK-NEXT:   hlcf.yield
    # CHECK-NEXT: }

    # CHECK-NEXT: %c = lit.var.decl
    # CHECK-NEXT: lifetime.start %c
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%c)
    var c = MemExample()
    # CHECK: hlcf.elif %cond {
    if cond:
        # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%c)
        # CHECK-NEXT: lifetime.end %c
        # CHECK-NEXT: lifetime.start %c
        # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%c)
        c = MemExample()
    # CHECK-NEXT:   hlcf.yield
    # CHECK-NEXT: } else {
    else:
        pass
    # CHECK-NEXT:   hlcf.yield
    # CHECK-NEXT: }
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %c
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])
    c.noop()
    # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%c)
    # CHECK-NEXT: lifetime.end %c

    # CHECK-NEXT: %d = lit.var.decl "d"
    # CHECK-NEXT: lifetime.start %d
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%d)
    var d = MemExample()

    # CHECK: [[ONE:%.+]] = kgen.param.constant: scalar<bool> = <true>
    # CHECK-NEXT: hlcf.elif [[ONE]] {
    # expected-warning @below {{'if' condition always evaluates to 'True'; 'else' branch is unreachable}}
    if True:
        # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %d
        # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])
        d.noop()
    # CHECK-NEXT:   hlcf.yield
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   kgen.unreachable
    # CHECK-NEXT: }

    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %d
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])
    d.noop()
    # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%d)
    # CHECK-NEXT: lifetime.end %d


# CHECK-LABEL: lit.fn @"try_examples
def try_examples(cond: __mlir_type.`!kgen.scalar<bool>`):
    # CHECK-NEXT: %a = lit.var.decl
    # CHECK-NEXT: %caught_error = lit.var.decl
    var a: MemExample
    # CHECK: lit.try "try0" {
    # CHECK-NOT: %a
    try:
        # The error value isn't used on the except branch, so it's copy from err
        # is completely optimized out.

        # CHECK-NEXT: lit.var.lifetime.start %caught_error
        # CHECK-NEXT: lit.call {{.*}}@Error::@"__init__{{.*}}(%caught_error)
        # CHECK-NEXT: lit.try.raise
        raise Error()
    # CHECK: } except {
    except caught_error:
        # CHECK-NEXT: lifetime.start %a
        # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%a)
        a = MemExample()
        # CHECK-NEXT: lit.ownership.use %caught_error
        # CHECK-NEXT: lit.ownership.use %caught_error
        _ = caught_error^
        # CHECK-NEXT: lit.call {{.*}}Error::@"__deinit__{{.*}}(%caught_error)
        # CHECK-NEXT: lit.var.lifetime.end %caught_error
        # CHECK-NEXT: lit.try.yield
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   kgen.unreachable
    # CHECK-NEXT: }
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %a
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])
    a.noop()  # ok
    # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%a)
    # CHECK-NEXT: lifetime.end %a

    # CHECK-NEXT: %_b = lit.var.decl
    var _b: MemExample
    # CHECK-NEXT: [[ERRSLOT:%.*]] = lit.var.decl "e"
    # CHECK-NEXT: lit.try "{{.*}}" {
    try:
        # CHECK-NEXT: lifetime.start %_b
        # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%_b)
        _b = MemExample()
        # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%_b)
        # CHECK-NEXT: lifetime.end %_b
        raise Error()
    # CHECK: } except {
    # CHECK-NEXT: [[ERR:%.*]] = lit.ref.immut [[ERRSLOT]]
    # CHECK-NEXT: lit.call {{.*}}use{{.*}}([[ERR]])

    # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}([[ERRSLOT]])
    # CHECK-NEXT: lit.var.lifetime.end [[ERRSLOT]]
    # CHECK-NEXT: lit.try.yield
    except e:
        use(e)
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   kgen.unreachable
    # CHECK-NEXT: }

    # CHECK-NEXT: lifetime.start %_b
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%_b)
    _b = MemExample()
    # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%_b)
    # CHECK-NEXT: lifetime.end %_b

    # CHECK-NEXT: %c = lit.var.decl
    var c: MemExample
    # CHECK-NEXT: [[ERRSLOT:%.*]] = lit.var.decl "e"
    # CHECK-NEXT: lit.try "{{.*}}" {
    try:
        # CHECK-NEXT: lifetime.start %c
        # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%c)
        c = MemExample()
        # CHECK-NOT: %c
        raise Error()
    # CHECK: } except {
    # CHECK-NEXT: [[ERR:%.*]] = lit.ref.immut [[ERRSLOT]]
    # CHECK-NEXT: lit.call {{.*}}use{{.*}}([[ERR]])

    # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}([[ERRSLOT]])
    # CHECK-NEXT: lifetime.end [[ERRSLOT]]

    # CHECK-NEXT: lit.try.yield
    except e:
        use(e)
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   kgen.unreachable
    # CHECK-NEXT: }
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %c
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])
    c.noop()
    # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%c)
    # CHECK-NEXT: lifetime.end %c

    # CHECK-NEXT: %d = lit.var.decl
    var d: MemExample
    # CHECK-NEXT: [[ERRSLOT:%.*]] = lit.var.decl "e"
    # CHECK-NEXT: lit.try "{{.*}}" {
    try:
        # CHECK-NEXT:  hlcf.elif %cond {
        if cond:
            raise Error()
        # CHECK-NOT: %d
    # CHECK: } except {
    except e:
        # CHECK: call {{.*}}Error::@"__deinit__
        use(e)
        # CHECK: lit.call {{.*}}__init__{{.*}}(%d)
        d = MemExample()
        # CHECK-NEXT: lit.try.yield
    # CHECK-NEXT: } else {
    else:
        # CHECK-NEXT: lifetime.start %d
        # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%d)
        d = MemExample()
        # CHECK-NEXT: lit.try.yield
    # CHECK-NEXT: }

    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %d
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])
    d.noop()
    # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%d)
    # CHECK-NEXT: lifetime.end %d


# CHECK-LABEL: lit.fn @"chris_origin_example
def chris_origin_example(a: Bool, b: Bool):
    var x: MemExample
    # CHECK: lit.try
    try:
        # CHECK: lit.try
        try:
            # CHECK: hlcf.elif
            if a:
                # CHECK: __init__{{.*}}(%x)
                x = MemExample()

                # CHECK: lit.try.raise
                raise Error()
        # CHECK: except
        # CHECK: lit.call {{.*}}__mlir_bool__
        # CHECK-NEXT: hlcf.elif %{{.*}} {
        # CHECK-NEXT: __deinit__{{.*}}(%x)
        # CHECK: return
        # CHECK: else
        # CHECK: lit.try.raise
        # CHECK: else
        # CHECK: hlcf.elif
        # CHECK: return
        # CHECK: else
        finally:
            if b:
                return
        # CHECK: } except {
    # CHECK: } except {
    except:
        # CHECK: lit.call {{.*}}@"use_mut{{.*}}(%x)
        # CHECK: lit.call {{.*}}__deinit__{{.*}}(%x)
        use_mut(x)
    # CHECK: else
    # CHECK-NEXT: lit.try.yield


# CHECK-LABEL: lit.fn @"loop_example
def loop_example(cond1: __mlir_type.`!kgen.scalar<bool>`, cond2: __mlir_type.`!kgen.scalar<bool>`):
    # CHECK-NEXT: %a = lit.var.decl "a"
    var a: MemExample
    # CHECK-NEXT: %b = lit.var.decl "b"
    var b: MemExample
    # CHECK-NEXT: %c = lit.var.decl "c"
    var c: MemExample

    # CHECK-NEXT: lifetime.start %a
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%a)
    a = MemExample()

    # Unneeded boilerplate due to 'while True':
    # CHECK-NEXT: hlcf.loop "_loop_0" {
    # CHECK-NEXT:  = kgen.param.constant: scalar<bool> = <true>
    # CHECK-NEXT:      hlcf.elif
    # CHECK-NEXT:        hlcf.yield
    # CHECK-NEXT:      } else {
    # CHECK-NEXT:        kgen.unreachable
    # CHECK-NEXT:      }
    while True:
        # CHECK-NEXT: lifetime.start %c
        # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%c)
        c = MemExample()
        # CHECK-NEXT: hlcf.elif %cond2 {
        if cond2:
            # CHECK-NEXT: lifetime.start %b
            # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%b)
            b = MemExample()
            # CHECK-NEXT: hlcf.break
            break
        # CHECK-NEXT: } else {
        # CHECK-NEXT:   lit.call {{.*}}__deinit__{{.*}}(%c)
        # CHECK-NEXT:   lifetime.end %c
        # CHECK-NEXT:   lit.call {{.*}}__deinit__{{.*}}(%a)
        # CHECK-NEXT:   lifetime.end %a
        # CHECK-NEXT:   hlcf.yield
        # CHECK-NEXT: }

        # CHECK-NEXT: lifetime.start %a
        # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%a)
        # CHECK-NEXT: hlcf.continue
        a = MemExample()
    # CHECK-NEXT: }

    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %a
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])
    a.noop()
    # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%a)
    # CHECK-NEXT: lifetime.end %a

    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %b
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])
    b.noop()
    # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%b)
    # CHECK-NEXT: lifetime.end %b

    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %c
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])
    c.noop()
    # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%c)
    # CHECK-NEXT: lifetime.end %c


# CHECK-LABEL: lit.struct.decl @TestLoopWithWholeObjectBit
struct TestLoopWithWholeObjectBit(Movable where False):
    var field: MemExample

    # CHECK: lit.fn @"__init__
    @implicit
    def __init__(out self, cond: __mlir_type.`!kgen.scalar<bool>`):
        # CHECK-NEXT: %buf = lit.var.decl "buf"
        # CHECK-NEXT: lifetime.start %buf
        # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%buf)
        var buf = MemExample()

        # CHECK-NEXT: hlcf.loop "_loop_0" {
        # CHECK-NEXT:   hlcf.elif %cond {
        # CHECK-NEXT:     hlcf.yield
        # CHECK-NEXT:   } else {
        # CHECK-NEXT:     hlcf.break
        # CHECK-NEXT:   }
        while cond:
            # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %buf
            # CHECK-NEXT:   lit.call {{.*}}noop{{.*}}([[IMMREF]])
            # CHECK-NEXT:   hlcf.continue
            buf.noop()
        # CHECK-NEXT: }

        # CHECK-NEXT: [[FIELD_REF:%.*]] = lit.ref.struct.ger %self[field]
        # CHECK-NEXT: lit.ownership.use %buf
        # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}move"
        # CHECK-NEXT: lifetime.end %buf
        # CHECK-NEXT: %none = kgen.param.constant
        # CHECK-NEXT: kgen.return
        self.field = buf^


# CHECK-LABEL: lit.fn @"testInfiniteloop
def testInfiniteloop():
    # CHECK-NEXT:  hlcf.loop "_loop_0" {
    # CHECK-NEXT:    [[T:%.+]] = kgen.param.constant: scalar<bool> = <true>
    # CHECK-NEXT:    hlcf.elif [[T]] {
    # CHECK-NEXT:      hlcf.yield
    # CHECK-NEXT:    } else {
    # CHECK-NEXT:      kgen.unreachable
    # CHECK-NEXT:    }
    while True:
        # CHECK-NEXT:  %localThing = lit.var.decl
        # CHECK-NEXT:  lifetime.start %localThing
        # CHECK-NEXT:  lit.call {{.*}}__init__{{.*}}(%localThing)
        # CHECK-NEXT:  [[IMMREF:%.*]] = lit.ref.immut %localThing
        # CHECK-NEXT:  lit.call {{.*}}noop{{.*}}([[IMMREF]])
        # CHECK-NEXT:  lit.call {{.*}}__deinit__{{.*}}(%localThing)
        # CHECK-NEXT:  lifetime.end %localThing
        var localThing = MemExample()
        localThing.noop()
    # CHECK-NEXT:      hlcf.continue
    # CHECK-NEXT:  }


@fieldwise_init
struct TrivialRange(Iterator, TrivialRegisterPassable):
    comptime Element = Int

    def __iter__(self) -> Self:
        return self

    def __next__(mut self) raises StopIteration -> Int:
        if self.__len__() > 0:
            return 1
        else:
            raise StopIteration()

    def __len__(self) -> Int:
        return 1


# Issue #98: https://github.com/modular/mojo/issues/98
# CHECK-LABEL: lit.fn @"mojo98
def mojo98(n: Int):
    var a = MemExample()
    for i in TrivialRange():
        a.x = i


struct MyStringReturningCtx(Movable):
    var s: String

    def __init__(out self):
        self.s = "hey"

    def __enter__(var self) -> Self:
        return self^

    def __init__(out self, *, deinit move: Self):
        self.s = ""

    def read(self) raises -> String:
        return ""


# CHECK-LABEL: lit.fn @"testErrorReturn
def testErrorReturn() raises:
    var input: String
    # CHECK: try
    with MyStringReturningCtx() as ctx:
        # CHECK-NOT: @MyStringReturningCtx::@"__deinit__
        var x = ctx.read()
        input = "hello"
    # CHECK: except
    use(input)


def marker():
    pass


# CHECK-LABEL: lit.fn @"test_param_for1
# MOCO-831
def test_param_for1(cond: Bool, cond2: Bool):
    # CHECK-NEXT: %mem = lit.var.decl
    # CHECK-NEXT: lifetime.start %mem
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%mem)
    var mem = MemExample()

    # CHECK: kgen.param.for [[ITER:[*].*]]: !TrivialRange in
    # CHECK-NEXT: has_next
    # CHECK-NEXT: get_next_iter
    # CHECK-SAME: {
    comptime for x in TrivialRange():
        # CHECK-NEXT: kgen.param.if {{.*}}paramfor_has_next

        # Make sure nothing sneaks in here.
        # CHECK-NEXT: lit.call {{.*}}marker()
        marker()

        # CHECK-NEXT: lit.call {{.*}}__mlir_bool__
        # CHECK-NEXT: hlcf.elif %{{.*}} {
        if cond:
            # CHECK: lit.call {{.*}}__deinit__{{.*}}(%mem)
            # CHECK-NEXT: lifetime.end %mem
            # CHECK-NEXT: lit.call {{.*}}marker()
            marker()
            # CHECK-NEXT: kgen.param.for.break
            break

        # CHECK: hlcf.elif %{{.*}} {
        if cond2:
            # CHECK: lit.call {{.*}}__deinit__{{.*}}(%mem)
            # CHECK-NEXT: lifetime.end %mem
            # CHECK-NEXT: lit.call {{.*}}marker()
            marker()
            # CHECK-NEXT: kgen.param.for.break
            break
        # CHECK-NEXT: } else {
        # CHECK-NEXT:   hlcf.yield
        # CHECK-NEXT: }
        # CHECK-NEXT: lit.call {{.*}}marker()
        marker()
        # CHECK-NEXT: kgen.param.for.continue

    # This is the else from the param.if inside the param.for.
    # CHECK-NEXT: } else {
    else:
        # CHECK-NEXT: [[TMP:%.*]] = lit.ref.immut %mem
        # CHECK-NEXT: lit.call {{.*}}use{{.*}}([[TMP]])
        use(mem)
        # CHECK: lit.call {{.*}}__deinit__{{.*}}(%mem)
        # CHECK-NEXT: lifetime.end %mem

        # CHECK-NEXT: lit.call {{.*}}marker()
        marker()
        # CHECK-NEXT: kgen.param.for.break
    # Finish out the param.if and the param.for
    # CHECK-NEXT:   } {elseIsolated, thenIsolated}
    # CHECK-NEXT:   kgen.unreachable
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   kgen.unreachable
    # CHECK-NEXT: } {bodyIsolated, elseIsolated}


# CHECK-LABEL: lit.fn @"test_param_for2
# MOCO-831
def test_param_for2():
    # CHECK: lit.call {{.*}}__init__{{.*}}(%mem)
    var mem = MemExample()

    # CHECK: kgen.param.for [[ITER:[*].*]]: !TrivialRange in
    # CHECK-NEXT: has_next
    # CHECK-NEXT: get_next_iter
    # CHECK-SAME: {
    comptime for x in TrivialRange():
        # CHECK-NEXT: kgen.param.if {{.*}}paramfor_has_next

        # Make sure nothing sneaks in here.
        # CHECK-NEXT: lit.call {{.*}}marker()
        marker()

        # CHECK-NEXT: lit.call {{.*}}use_mut{{.*}}(%mem)
        use_mut(mem)
        # CHECK-NEXT: kgen.param.for.continue

    # CHECK-NEXT: } else {
    else:
        # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%mem)
        # CHECK-NEXT: lifetime.end %mem
        # CHECK-NEXT: lit.call {{.*}}marker()
        marker()

        # CHECK-NEXT: kgen.param.for.break


# CHECK-LABEL: lit.fn @"test_elif
def test_elif(cond: Bool, cond2: Bool):
    var mem1 = MemExample()
    var mem2 = MemExample()
    var mem3 = MemExample()

    # CHECK:  __mlir_bool__
    # CHECK-NEXT: hlcf.elif %{{.*}} {
    if cond:
        # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%mem3)
        # CHECK-NEXT: lifetime.end %mem3
        # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%mem2)
        # CHECK-NEXT: lifetime.end %mem2
        # CHECK-NEXT: lit.call {{.*}}use_mut{{.*}}(%mem1)
        use_mut(mem1)
        # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%mem1)
        # CHECK-NEXT: lifetime.end %mem1

        # CHECK-NEXT: lit.call {{.*}}marker()
        marker()
        # CHECK-NEXT: hlcf.yield
    # CHECK-NEXT: } else {
    # mem1 never used at this point, destroy in the condition.
    # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%mem1)
    # CHECK-NEXT: lifetime.end %mem1
    # CHECK-NEXT: __mlir_bool__
    # CHECK-NEXT: hlcf.elif.yield

    # CHECK-NEXT: } then {
    elif cond2:
        # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%mem3)
        # CHECK-NEXT: lifetime.end %mem3
        # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%mem2)
        # CHECK-NEXT: lifetime.end %mem2

        # CHECK-NEXT: lit.call {{.*}}marker()
        marker()
        # CHECK-NEXT: hlcf.yield
    # CHECK-NEXT: } else {

    # Last use of mem2 is in this condition.
    # CHECK-NEXT: lit.ref.struct.ger %mem2[x]
    # CHECK-NEXT: kgen.rebind
    # CHECK-NEXT: lit.ref.load
    # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%mem2)
    # CHECK: hlcf.elif.yield

    # CHECK-NEXT: } then {
    elif mem2.x == 0:
        # CHECK-NEXT: lit.call {{.*}}use_mut{{.*}}(%mem3)
        use_mut(mem3)
        # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%mem3)
        # CHECK-NEXT: lifetime.end %mem3

        # CHECK-NEXT: lit.call {{.*}}marker()
        marker()
        # CHECK-NEXT: hlcf.yield

    # CHECK-NEXT: } else {
    # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%mem3)
    # CHECK-NEXT: lifetime.end %mem3
    # CHECK-NEXT: hlcf.yield


# https://github.com/modular/mojo/issues/3710
# Mojo frees memory while reference to it is still in use
# CHECK-LABEL: lit.fn @"loop_any_origin
def loop_any_origin(var mem: MemExample, cond: Bool):
    # CHECK: lit.call {{.*}}unsafe_ptr
    var ptr = mem.unsafe_ptr()

    # The "mem" destructor must be in the loop exit, not ahead of the loop because
    # there is an access through AnyOrigin within the loop.
    # CHECK: hlcf.loop
    # CHECK-NEXT:     lit.call {{.*}}Bool::@"__mlir_bool__
    # CHECK-NEXT:     hlcf.elif
    # CHECK-NEXT:       hlcf.yield
    # CHECK-NEXT:     } else {
    # CHECK-NEXT:       lit.call {{.*}}MemExample::@"__deinit__
    # CHECK-NEXT:       lit.var.lifetime.end %ptr
    # CHECK-NEXT:       hlcf.break
    while cond:
        ptr[] = 4


# 4694: or/and handling of comparisons on PythonObject
struct PyObjLike(RegisterPassable):
    def __init__(out self, *, copy: Self):
        pass

    def __eq__(self, other: Self) raises -> Self:
        while True:
            pass

    def __bool__(self) raises -> Bool:
        return True

    def __deinit__(deinit self):
        pass


# CHECK-LABEL: lit.fn @"test4694
def test4694(a: PyObjLike, b: PyObjLike) raises:
    # Just check that we don't get a verifier error.
    if a == b or b == a:
        gotit()


def gotit():
    pass
