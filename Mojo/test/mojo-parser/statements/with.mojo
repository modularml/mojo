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

# RUN: %parse-mojo-isolated %s -verify-diagnostics | FileCheck %s

##===----------------------------------------------------------------------===##
# With
##===----------------------------------------------------------------------===##


# Issue #12358
# CHECK-LABEL: lit.fn @"raise_error
def raise_error() raises:
    # CHECK-NEXT: [[TMP:%.*]] = kgen.param.constant: {{.*}}#StringLiteral <:string "thing">> = <{}>
    # CHECK-NEXT: [[ERR:%.*]] = lit.call {{.*}}Error::@"__init__{{.*}}([[TMP]], %__error__)
    # CHECK-NEXT: lit.raise
    raise "thing"


def raise_string() raises String:
    pass


def raise_int() raises Int:
    pass


struct ExampleCM(ImplicitlyCopyable):
    def __enter__(self) -> Int:
        return 42

    def __exit__(self):
        pass  # normal

    def __exit__(self, err: Error) -> Bool:
        return True  # Raise


# Cannot use mutating __enter__
# https://github.com/modularml/modular/issues/27371
struct MutatingCM(Movable where False):
    def __init__(out self):
        pass

    def __enter__(mut self) -> Int:
        return 42

    def __exit__(mut self):
        pass  # normal


@fieldwise_init
struct NoExitCMReg(Movable where False):
    def __enter__(mut self) -> Int:
        pass


@fieldwise_init
struct NoExitCMMem(Movable where False):
    def __enter__(mut self) -> Self:
        pass


def noop(a: Int):
    pass


# CHECK-LABEL: lit.fn @"testWithNonRaising
def testWithNonRaising(a: ExampleCM):
    # CHECK-NEXT: %$CONTEXTMGR = lit.var.decl "$CONTEXTMGR"
    # CHECK-NEXT: lit.memcpy %a, %$CONTEXTMGR
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %$CONTEXTMGR
    # CHECK-NEXT: [[TARGET:%.*]] = lit.call {{.*}}__enter__{{.*}}([[IMMREF]])
    # CHECK-NEXT: %val = lit.var.decl "val"
    # CHECK-NEXT: lit.ref.store [[TARGET]], %val
    # CHECK-NEXT: %__with_error__
    # CHECK-NEXT: lit.try %__with_error__
    with a as val:
        # CHECK-NEXT: [[VALR:%.*]] = kgen.rebind %val
        # CHECK-NEXT: [[VAL:%.*]] = lit.ref.load [[VALR]]
        # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[VAL]])
        noop(val)
    # CHECK: finally
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %$CONTEXTMGR
    # CHECK-NEXT: lit.call {{.*}}__exit__{{.*}}([[IMMREF]])

    # Test a with with no target.

    # CHECK: %$CONTEXTMGR_0 = lit.var.decl "$CONTEXTMGR"
    # CHECK-NEXT: lit.memcpy %a, %$CONTEXTMGR_0
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %$CONTEXTMGR_0
    # CHECK: lit.call {{.*}}__enter__{{.*}}([[IMMREF]]
    # CHECK: lit.try
    with a:
        # CHECK-NEXT: kgen.param.constant: {{.*}}42
        # CHECK-NEXT: lit.call {{.*}}noop
        noop(42)
    # CHECK: finally
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %$CONTEXTMGR_0
    # CHECK-NEXT: lit.call {{.*}}__exit__{{.*}}([[IMMREF]])

    # CHECK: [[MGR:%.*]] = lit.var.decl "$CONTEXTMGR"{{.*}}!MutatingCM
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}([[MGR]])
    # CHECK-NEXT: lit.call {{.*}}__enter__{{.*}}([[MGR]])
    # CHECK-NEXT: %val{{.*}} = lit.var.decl "val"
    with MutatingCM() as val:
        # CHECK: lit.call {{.*}}noop
        noop(val)
    # CHECK: lit.call {{.*}}__exit__{{.*}}([[MGR]])

    # CHECK: [[REG:%.*]] = lit.call {{.*}}NoExitCMReg::@"__enter__
    with NoExitCMReg():
        pass
    # CHECK: finally
    # CHECK-NEXT: ownership.use [[REG]]

    # CHECK: call {{.*}}NoExitCMMem::@"__enter__{{.*}}(%{{.*}}, [[MEM:%.*]]) : !lit.generator
    with NoExitCMMem():
        pass
    # CHECK: finally
    # CHECK-NEXT: ownership.use [[MEM]]


# CHECK-LABEL: lit.fn @"testWithRaising
def testWithRaising(a: ExampleCM) raises:
    # CHECK: %$CONTEXTMGR = lit.var.decl
    # CHECK-NEXT: lit.memcpy %a, %$CONTEXTMGR
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %$CONTEXTMGR
    # CHECK-NEXT: [[TARGET:%.*]] = lit.call {{.*}}__enter__{{.*}}([[IMMREF]])
    # CHECK: %val = lit.var.decl "val"
    # CHECK-NEXT: lit.ref.store [[TARGET]], %val
    # CHECK-NEXT: %__with_error__ = lit.var.decl
    # CHECK: lit.try %__with_error__
    # CHECK: lit.try %__inner_error__
    with a as val:
        # CHECK-NEXT: [[VALR:%.*]] = kgen.rebind %val
        # CHECK-NEXT: [[VAL:%.*]] = lit.ref.load [[VALR]]
        # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[VAL]])
        noop(val)

        # CHECK: [[RESULT:%.*]] = lit.call {{.*}}raise_error{{.*}}(%__inner_error__, %__call_result_tmp__
        raise_error()
        # CHECK-NEXT: lit.try.yield
    # CHECK-NEXT: } except {
    # CHECK:        [[FALSE:%.*]] = kgen.param.constant: scalar<bool> = <false>
    # CHECK-NEXT:   lit.ref.store [[FALSE]], %__with_exc__
    # CHECK-NEXT:   [[IMMREF:%.*]] = lit.ref.immut %$CONTEXTMGR
    # CHECK-NEXT:   [[ERROR:%.*]] = lit.ref.immut %__inner_error__
    # CHECK-NEXT:   [[EXIT_RESULT:%.*]] = lit.call {{.*}}__exit__{{.*}}([[IMMREF]], [[ERROR]])
    # CHECK-NEXT:   [[SUCCESS_SB:%.*]] = lit.call {{.*}}__mlir_bool__{{.*}}([[EXIT_RESULT]])
    # CHECK-NEXT:   hlcf.elif [[SUCCESS_SB]] {
    # CHECK-NEXT:     hlcf.yield
    # CHECK-NEXT:   } else {
    # CHECK-NEXT:     lit.call {{.*}}Error::@"__init__{{.*}}"{{.*}}(%__inner_error__, %__with_error__){{.*}}*, "move"
    # CHECK-NEXT:     lit.raise
    # CHECK-NEXT:     hlcf.yield
    # CHECK-NEXT:   }
    # CHECK-NEXT:   lit.try.yield
    # CHECK:      } finally {
    # CHECK:    } except {
    # CHECK-NEXT: lit.call {{.*}}Error::@"__init__{{.*}}"{{.*}}(%__with_error__, %__error__){{.*}}*, "move"
    # CHECK:    } finally {
    # CHECK-NEXT: %__finally_error__ = lit.var.decl
    # CHECK-NEXT: lit.try
    # CHECK-NEXT:   %[[EXC:.*]] = lit.ref.load %__with_exc__
    # CHECK-NEXT:   hlcf.elif %[[EXC]]
    # CHECK-NEXT:   [[IMMREF:%.*]] = lit.ref.immut %$CONTEXTMGR
    # CHECK-NEXT:   call {{.*}}__exit__{{.*}}([[IMMREF]])


# CHECK-LABEL: lit.fn @"testWithInTry
def testWithInTry(a: ExampleCM):
    # CHECK: %e = lit.var.decl "e" var
    # CHECK-NEXT: lit.try %e
    try:
        # CHECK: %$CONTEXTMGR = lit.var.decl
        # CHECK-NEXT: lit.memcpy %a, %$CONTEXTMGR
        # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %$CONTEXTMGR
        # CHECK-NEXT: [[TARGET:%.*]] = lit.call {{.*}}__enter__{{.*}}([[IMMREF]])
        # CHECK: %cm = lit.var.decl "cm"
        # CHECK-NEXT: lit.ref.store [[TARGET]], %cm
        # CHECK: [[TRUE:%.*]] = kgen.param.constant: scalar<bool> = <true>
        # CHECK-NEXT: lit.ref.store [[TRUE]], %__with_exc__
        # CHECK-NEXT: lit.try %__with_error__
        with a as cm:
            # CHECK: %__inner_error__ = lit.var.decl
            # CHECK: lit.try %__inner_error__
            # CHECK: [[RESULT:%.*]] = lit.call {{.*}}raise_error{{.*}}(%__inner_error__, %__call_result_tmp__
            raise_error()
    except e:
        _ = e


# CHECK-LABEL: lit.fn @"testWithScoping
def testWithScoping(a: ExampleCM):
    # This is a test that issue #18811 is fixed, in which a `with`
    # statement inside a `def` does not respect lexical scope and binds
    # its variable in its parent scope.
    with a as withDecl:
        # CHECK: %withDecl = lit.var.decl "withDecl"
        noop(withDecl)
    with a as withDecl:
        # CHECK: = lit.var.decl "withDecl"
        noop(withDecl)


# CHECK-LABEL: lit.fn @"testWithInDef
def testWithInDef(a: ExampleCM) raises:
    # This is a test that issue #20141 is fixed.
    # https://github.com/modularml/modular/issues/20141
    # IE that when used inside a `def`, the `with` statement uses
    # mutable function scope variables.
    # CHECK: [[VAL1R:%.*]] = kgen.rebind %val1
    # CHECK: [[VAL1:%.*]] = lit.ref.load [[VAL1R]]
    var val1 = 77
    # CHECK: lit.call {{.*}}noop{{.*}}([[VAL1]])
    noop(val1)
    with a as val1:
        # CHECK: [[VAL1R:%.*]] = kgen.rebind %val1
        # CHECK-NEXT: [[VAL1:%.*]] = lit.ref.load [[VAL1R]]
        # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[VAL1]])
        noop(val1)
    noop(val1)
    with a as val2:
        # CHECK: [[VAL2R:%.*]] = kgen.rebind %val2
        # CHECK-NEXT: [[VAL2:%.*]] = lit.ref.load [[VAL2R]]
        # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[VAL2]])
        noop(val2)
    # CHECK: [[VAL2R:%.*]] = kgen.rebind %val2
    # CHECK: [[VAL2:%.*]] = lit.ref.load [[VAL2R]]
    var val2 = 78
    # CHECK: lit.call {{.*}}noop{{.*}}([[VAL2]])
    noop(val2)


# Issue #21990: [Mojo-lang] Support context managers in with statements that
# don't implement the __exit__ method.
# https://github.com/modularml/modular/issues/21990


struct CMWithoutExit(Movable):
    def __init__(out self):
        pass

    # This context manager consumes itself and returns it as the value.
    def __enter__(var self) -> Self:
        return self^

    def method(self):
        pass


# CHECK-LABEL: lit.fn @"testCMWithoutExit
def testCMWithoutExit():
    # CHECK: %$CONTEXTMGR = lit.var.decl "$CONTEXTMGR"
    # CHECK: %a = lit.var.decl "a"
    # CHECK: lit.call {{.*}}@CMWithoutExit::@"__enter__{{.*}}(%$CONTEXTMGR, %a)
    # CHECK-NEXT: %__with_error__ = lit.var.decl "__with_error__" synth
    # CHECK-NEXT: lit.try %__with_error__
    # CHECK-NEXT:   [[IMMREF:%.*]] = lit.ref.immut %a
    # CHECK-NEXT:   lit.call {{.*}}@CMWithoutExit::@"method{{.*}}([[IMMREF]])
    # CHECK-NEXT:   lit.try.yield
    # CHECK-NEXT: } except {
    # CHECK-NEXT:   kgen.unreachable
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   lit.try.yield
    # CHECK-NEXT: } finally {
    # CHECK-NEXT:   lit.ownership.use %a
    # CHECK-NEXT:   lit.try.yield
    # CHECK-NEXT: }
    with CMWithoutExit() as a:
        a.method()

    # CHECK: %$CONTEXTMGR_0 = lit.var.decl "$CONTEXTMGR"
    # CHECK-NEXT: lit.call {{.*}}@CMWithoutExit::@"__init__{{.*}}(%$CONTEXTMGR_0)
    # CHECK: %a_1 = lit.var.decl "a"
    # CHECK-NEXT: lit.call {{.*}}@CMWithoutExit::@"__enter__{{.*}}(%$CONTEXTMGR_0, %a_1)
    # CHECK-NEXT: %__with_error__{{.*}} = lit.var.decl "__with_error__" synth
    # CHECK-NEXT: lit.try %__with_error__
    # CHECK-NEXT:   [[IMMREF:%.*]] = lit.ref.immut %a_1
    # CHECK-NEXT:   lit.call {{.*}}@CMWithoutExit::@"method{{.*}}([[IMMREF]])
    # CHECK-NEXT:   lit.try.yield
    # CHECK-NEXT: } except {
    # CHECK-NEXT:   kgen.unreachable
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   lit.try.yield
    # CHECK-NEXT: } finally {
    # CHECK-NEXT:   lit.ownership.use %a_1
    # CHECK-NEXT:   lit.try.yield
    # CHECK-NEXT: }

    # Test that we don't have a name collision between two 'a's.
    with CMWithoutExit() as a:
        a.method()

    # Test that we can nest with statements.
    with CMWithoutExit() as a:
        with CMWithoutExit() as b:
            b.method()


# CHECK-LABEL: lit.fn @"testMultiClauseWith
def testMultiClauseWith():
    # Tests that multiple-clause `with` statements (like below) are interpreted
    # as multiple nested "single" with statements like.
    #     with MyClass() as a:
    #         with MyClass() as b:
    #             ...

    # CHECK: [[CMA:%.*]] = lit.var.decl "$CONTEXTMGR"
    # CHECK-NEXT: lit.call {{.*}}@CMWithoutExit::@"__init__{{.*}}([[CMA]])
    # CHECK: %a = lit.var.decl "a"
    # CHECK-NEXT: lit.call {{.*}}@CMWithoutExit::@"__enter__{{.*}}([[CMA]], %a)
    # CHECK-NEXT: %__with_error__{{.*}} = lit.var.decl "__with_error__" synth
    # CHECK-NEXT: lit.try %__with_error__
    #
    # CHECK: [[CMB:%.*]] = lit.var.decl "$CONTEXTMGR"
    # CHECK-NEXT: lit.call {{.*}}@CMWithoutExit::@"__init__{{.*}}([[CMB]])
    # CHECK: %b = lit.var.decl "b"
    # CHECK-NEXT: lit.call {{.*}}@CMWithoutExit::@"__enter__{{.*}}([[CMB]], %b)
    # CHECK-NEXT: %__with_error__{{.*}} = lit.var.decl "__with_error__" synth
    # CHECK-NEXT: lit.try %__with_error__
    with CMWithoutExit() as a, CMWithoutExit() as b:
        # CHECK:   lit.call {{.*}}@CMWithoutExit::@"method{{.*}}
        a.method()
        b.method()

    # Make sure that we destroy them in the right order.
    # CHECK:   lit.ownership.use %b
    # CHECK:   lit.ownership.use %a


# CHECK-LABEL: lit.fn @"testAmbiguousMultiContextWith
def testAmbiguousMultiContextWith():
    # Make sure that we don't interpret the below like this:
    #     with (CMWithoutExit(), CMWithoutExit()) as b:
    #         noop(b)
    # In other words, we don't want to put a tuple into b.
    # If this compiles it all, it should work, because a tuple would be
    # rejected as it doesn't have an __enter__.
    with CMWithoutExit(), CMWithoutExit() as b:
        pass


# CHECK-LABEL: lit.fn @"testCMWithoutExitEarlyReturn
# https://github.com/modularml/modular/issues/23693
def testCMWithoutExitEarlyReturn():
    # CHECK: %$CONTEXTMGR = lit.var.decl "$CONTEXTMGR"
    # CHECK-NEXT: lit.call {{.*}}@CMWithoutExit::@"__init__{{.*}}(%$CONTEXTMGR)
    # CHECK: %a = lit.var.decl "a"
    # CHECK-NEXT: lit.call {{.*}}@CMWithoutExit::@"__enter__{{.*}}(%$CONTEXTMGR, %a)
    # CHECK-NEXT: %__with_error__ = lit.var.decl "__with_error__" synth
    # CHECK-NEXT: lit.try %__with_error__
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %a
    # CHECK-NEXT:   lit.call {{.*}}@CMWithoutExit::@"method{{.*}}([[IMMREF]])
    # CHECK-NEXT:   kgen.param.constant: none
    # CHECK-NEXT:   lit.return
    # CHECK-NEXT:   lit.try.yield
    # CHECK-NEXT: } except {
    # CHECK-NEXT:   kgen.unreachable
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   lit.try.yield
    # CHECK-NEXT: } finally {
    # CHECK-NEXT:   lit.ownership.use %a
    # CHECK-NEXT:   lit.try.yield
    # CHECK-NEXT: }
    with CMWithoutExit() as a:
        a.method()
        return


@fieldwise_init
struct CMUnconditionalExit(Movable where False):
    def __enter__(self):
        pass

    def __exit__(mut self):
        pass


# CHECK-LABEL: lit.fn @"unconditional_exit
def unconditional_exit() raises:
    # CHECK: lit.try %__with_error__
    with CMUnconditionalExit():
        # CHECK: lit.call {{.*}}noop
        # CHECK-NEXT:  lit.try.yield
        noop(42)
    # CHECK-NEXT: } except {
    # CHECK-NEXT:   lit.call {{.*}}Error::@"__init__{{.*}}"{{.*}}(%__with_error__, %__error__){{.*}}*, "move"
    # CHECK-NEXT:   lit.raise
    # CHECK-NEXT:   lit.try.yield
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   lit.try.yield
    # CHECK-NEXT: finally {
    # CHECK-NEXT: %__finally_error__ = lit.var.decl
    # CHECK-NEXT:   lit.try %__finally_error__
    # CHECK:          call {{.*}}__exit__{{.*}}(%$CONTEXTMGR)
    # CHECK:        } except {
    # CHECK-NEXT:     kgen.unreachable


struct ExampleCMTuple(ImplicitlyCopyable):
    def __init__(out self, *, copy: Self):
        pass

    def __enter__(self) -> Tuple[Int, Int]:
        return (42, 43)

    def __exit__(self):
        pass  # normal

    def __exit__(self, err: Error) -> Bool:
        return True  # Raise


# CHECK-LABEL: lit.fn @"testExampleCMTuple
def testExampleCMTuple(cm: ExampleCMTuple):
    # CHECK: %a = lit.var.decl "a"
    # CHECK: %b = lit.var.decl "b"
    # CHECK: [[TARGET:%.*]] = lit.call {{.*}}__enter__
    with cm as (a, b):
        # Captures the refs from the tuple.
        # CHECK: [[A:%.*]] = lit.ref.load %a
        # CHECK: [[B:%.*]] = lit.ref.load %b
        # CHECK: lit.call {{.*}}@SIMD::@"__add__{{.*}}([[A]], [[B]])
        _ = a + b


# A context manager that has a generic __exit__ method.
struct GenericExitCtxtMgr(Movable where False):
    var handle: Bool

    @implicit
    def __init__(out self, handle: Bool = True):
        self.handle = handle

    def __enter__(self):
        pass

    def __exit__(self):
        pass

    def __exit__[ErrType: AnyType](self, err: ErrType) -> Bool:
        return self.handle


def with_infer_error() raises:
    with GenericExitCtxtMgr():
        raise_error()


def with_infer_string() raises String:
    with GenericExitCtxtMgr():
        raise_string()


def with_infer_int() raises Int:
    with GenericExitCtxtMgr():
        raise_int()


# A context manager that has a Float __exit__ method.
struct FloatErrorExitCtxtMgr(Movable where False):
    var handle: Bool

    @implicit
    def __init__(out self, handle: Bool = True):
        self.handle = handle

    def __enter__(self):
        pass

    def __exit__(self):
        pass

    def __exit__(self, err: Float32) -> Bool:
        return self.handle


def with_impl_convert() raises Float32:
    with FloatErrorExitCtxtMgr():
        # This should implicitly convert to Float32, not be a type error.
        raise_int()


def with_doesnt_actually_throw():
    with FloatErrorExitCtxtMgr():
        noop(42)


# Issue #5176: Context manager don't work with consuming __exit__(var self)
@fieldwise_init
struct ConsumingExitCM(Movable):
    def __enter__(mut self):
        pass

    def __exit__(var self):
        pass


def testConsumingExitCM():
    with ConsumingExitCM() as a:
        _ = a
