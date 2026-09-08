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
# RUN: %parse-mojo-isolated %s --debug-level full -o /dev/null

# Error Handling related CheckLifetimes tests.


def use(x: Int):
    pass


def use(x: String):
    pass


def use_and_raise(x: Int) raises:
    pass


# CHECK-LABEL: lit.struct.decl @RegExample
struct RegExample(ImplicitlyCopyable, RegisterPassable):
    def __init__(out self):
        return

    # CHECK: lit.fn @"__init__{{.*}}"{{.*}}(*, %copy:
    def __init__(out self, *, copy: Self):
        return

    # Test a raising constructor.
    # CHECK-LABEL: lit.fn @"__init__{{.*}}(%a: {{.*}}, %b: {{.*}}, ?, %__error__: !lit.ref<!Error, {{.*}}> byref_error, %self: !lit.ref<!RegExample, {{.*}}> byref_result) throws -> !kgen.scalar<bool>
    def __init__(out self, a: MemExample, b: MemExample) raises:
        # CHECK-NOT: __deinit__
        # CHECK: [[FALSE:%.*]] = kgen.param.constant: scalar<bool> = <false>
        # CHECK-NEXT: kgen.return [[FALSE]]
        return

    def noop(self):
        pass

    def __deinit__(deinit self):
        pass

    def mutate(mut self):
        pass


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

    def __deinit__(deinit self):
        pass


# Use of uninitialized value after call to def function


# CHECK-LABEL: lit.fn @"error_handling_int_let
# https://github.com/modularml/modular/issues/25419
def error_handling_int_let() raises:
    # CHECK: lit.var.decl "x"
    var x: Int = 1
    _ = use_and_raise(x)
    use(x)


def somethingThatRaises() raises:
    pass


# CHECK-LABEL: lit.fn @"thing_that_raises
def thing_that_raises(c: __mlir_type.`!kgen.scalar<bool>`) raises -> MemExample:
    # CHECK-NEXT: [[RESULT:%.*]] = lit.var.decl "__call_result_tmp__" synth : !lit.ref<none,
    # CHECK-NEXT: lifetime.start [[RESULT]]
    # CHECK-NEXT: [[IS_ERR:%.*]] = lit.call {{.*}}somethingThatRaises{{.*}}(%__error__, [[RESULT]])
    # CHECK-NEXT: hlcf.if [[IS_ERR]]
    # CHECK-NEXT:   mark_consumed [[RESULT]]
    # CHECK-NEXT:   lifetime.end [[RESULT]]
    # CHECK-NEXT:   [[TRUE:%.*]] = kgen.param.constant: scalar<bool> = <true>
    # CHECK-NEXT:   lit.error_return [[TRUE]]
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   lifetime.end [[RESULT]]
    # CHECK-NEXT:   mark_consumed %__error__
    # CHECK-NEXT:   yield
    # CHECK-NEXT: }
    somethingThatRaises()

    # CHECK-NEXT:   hlcf.elif %c {
    # CHECK-NEXT:      = lit.call {{.*}}__init__
    # CHECK-NOT:       __deinit__
    # CHECK: kgen.return
    if c:
        return MemExample()
    # CHECK-NEXT:  } else {
    raise Error("TypeError: cannot invert values of this type")


struct RaisingInit(Movable where False):
    var stream: Int

    def __init__(out self, flags: Int = 0) raises:
        var stream = 4
        # This can raise, but 'self' doesn't need to be initialized.
        _ = somethingThatRaises()
        self.stream = stream


# CHECK-LABEL: lit.fn @"finally_may_raise
def finally_may_raise() raises:
    # CHECK: lit.try
    try:
        # CHECK-NEXT: lifetime.start %__try_error__
        # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}(%__try_error__)
        # CHECK-NEXT: lit.try.raise
        raise Error()
        # CHECK-NEXT: } except {
        # CHECK-NEXT: lit.call {{.*}}Error::@"__init__{{.*}}"{{.*}}(%__try_error__, %__error__){{.*}}*, "move"
        # CHECK-NEXT: lit.var.lifetime.end %__try_error__
        # CHECK-NEXT: [[TRUE:%.*]] = kgen.param.constant: scalar<bool> = <true>
        # CHECK-NEXT: %__finally_error__ = lit.var.decl
        # CHECK-NEXT: lit.try
        # CHECK-NEXT:   [[RESULT:%.*]] = lit.var.decl
        # CHECK-NEXT:   lifetime.start %__finally_error__
        # CHECK-NEXT:   lifetime.start [[RESULT]]
        # CHECK-NEXT:   [[IS_ERR:%.*]] = lit.call {{.*}}somethingThatRaises{{.*}}(%__finally_error__, [[RESULT]])
        # CHECK-NEXT:   if [[IS_ERR]]
        # CHECK-NEXT:     call {{.*}}__deinit__{{.*}}(%__error__)
        # CHECK-NEXT:     mark_consumed [[RESULT]]
        # CHECK:      } except {
        # CHECK-NEXT:   lit.call {{.*}}Error::@"__init__{{.*}}"{{.*}}(%__finally_error__, %__error__){{.*}}*, "move"
        # CHECK-NEXT:   lit.var.lifetime.end %__finally_error__
        # CHECK:      else
        # CHECK-NEXT:   lit.try.yield
        # CHECK-NEXT: }
        # CHECK-NEXT: lit.error_return [[TRUE]]
    finally:
        somethingThatRaises()


@fieldwise_init
struct ThrowingExit(Movable where False):
    def __enter__(self):
        pass

    def __deinit__(deinit self):
        pass

    def __exit__(self) raises:
        pass

    def __exit__(self, e: Error) raises -> Bool:
        return False


# CHECK-LABEL: lit.fn @"context_mgr_exit_raises
def context_mgr_exit_raises() raises:
    # CHECK: lit.call {{.*}}Error::@"__init__{{.*}}"{{.*}}(%__with_error__, %__error__){{.*}}*, "move"
    # CHECK-NEXT: lit.var.lifetime.end %__with_error__
    # CHECK-NEXT: [[TRUE:%.*]] = kgen.param.constant: scalar<bool> = <true>
    # CHECK-NEXT: %__finally_error__ = lit.var.decl
    # CHECK-NEXT: lit.try
    # CHECK-NEXT:   [[DID_ERR:%.*]] = lit.ref.load %__with_exc__
    # CHECK-NEXT:   lifetime.end %__with_exc__
    # CHECK-NEXT:   hlcf.elif [[DID_ERR]]
    # CHECK-NEXT:     [[IMM:%.*]] = lit.ref.immut %$CONTEXTMGR
    # CHECK-NEXT:     [[BOOL:%.*]] = lit.var.decl
    # CHECK-NEXT:     lit.var.lifetime.start %__finally_error__
    # CHECK-NEXT:     lit.var.lifetime.start [[BOOL]]
    # CHECK-NEXT:     [[IS_ERR:%.*]] = lit.call {{.*}}__exit__{{.*}}([[IMM]], %__finally_error__, [[BOOL]])
    # CHECK-NEXT:     call {{.*}}__deinit__{{.*}}(%$CONTEXTMGR)
    # CHECK-NEXT:     lit.var.lifetime.end %$CONTEXTMGR
    # CHECK-NEXT:     if [[IS_ERR]]
    # CHECK-NEXT:       call {{.*}}__deinit__{{.*}}(%__error__)
    # CHECK-NEXT:       mark_consumed [[BOOL]]
    # CHECK-NEXT:       lit.var.lifetime.end [[BOOL]]
    # CHECK:          else
    # CHECK-NEXT:       lit.var.lifetime.end [[BOOL]]
    # CHECK-NEXT:       mark_consumed %__finally_error__
    # CHECK-NEXT:       lit.var.lifetime.end %__finally_error__
    # CHECK-NEXT:       yield
    # CHECK:        else
    # CHECK-NEXT:     call {{.*}}__deinit__{{.*}}(%$CONTEXTMGR)
    # CHECK:      } except {
    # CHECK-NEXT:   lit.call {{.*}}Error::@"__init__{{.*}}"{{.*}}(%__finally_error__, %__error__){{.*}}*, "move"
    # CHECK-NEXT:   lit.var.lifetime.end %__finally_error__
    # CHECK:      else
    # CHECK-NEXT:   lit.try.yield
    # CHECK-NEXT: }
    # CHECK-NEXT: lit.error_return [[TRUE]]
    with ThrowingExit():
        raise Error()


def may_throw() raises -> RegExample:
    return RegExample()


# CHECK-LABEL: lit.fn @"propagate_reg_error
def propagate_reg_error() raises:
    # CHECK-NEXT: [[RESULT:%.*]] = lit.var.decl "__call_result_tmp__" synth : !lit.ref<!RegExample,
    # CHECK-NEXT: lifetime.start [[RESULT]]
    # CHECK-NEXT: %0 = lit.call {{.*}}may_throw{{.*}}(%__error__, [[RESULT]])
    # CHECK-NEXT: if %0
    # CHECK:        lifetime.end [[RESULT]]
    # CHECK:        lit.error_return
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   mark_consumed %__error__
    # CHECK-NEXT:   yield
    # CHECK-NEXT: }
    # CHECK-NEXT: lit.ownership.use [[RESULT]]
    # CHECK-NEXT: lit.call {{.*}}@RegExample::@"__deinit__{{.*}}([[RESULT]])
    # CHECK-NEXT: lifetime.end [[RESULT]]
    _ = may_throw()
    # CHECK-NEXT: %none = kgen.param.constant: none
    # CHECK-NEXT: lit.ref.store %none, %__result__
    # CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: scalar<bool> = <false>
    # CHECK-NEXT: kgen.return [[FALSE]]


# CHECK-LABEL: lit.struct.decl @BigRegExample
struct BigRegExample(RegisterPassable):
    var a: RegExample
    var b: RegExample

    # Test a raising constructor.
    # CHECK-LABEL: lit.fn @"__init__{{.*}}MemExample{{.*}}MemExample
    def __init__(out self, a: MemExample, b: MemExample) raises:
        # CHECK-NEXT: [[A_REF:%.*]] = lit.ref.struct.ger %self[a]
        # CHECK-NEXT: [[A:%.*]] = lit.call {{.*}}__init__{{.*}}()
        # CHECK-NEXT: lit.ref.store [[A]], [[A_REF]]
        self.a = RegExample()
        # CHECK-NEXT: [[B_REF:%.*]] = lit.ref.struct.ger %self[b]
        # CHECK-NEXT: [[B:%.*]] = lit.call {{.*}}__init__{{.*}}()
        # CHECK-NEXT: lit.ref.store [[B]], [[B_REF]]
        self.b = RegExample()
        # CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: scalar<bool> = <false>
        # CHECK-NEXT: kgen.return [[FALSE]]


struct MyStringReturningCtx(Movable):
    var s: String

    def __init__(out self):
        self.s = "hey"

    def __enter__(var self) -> Self:
        return self^

    def read(self) raises -> String:
        return ""


# CHECK: lit.fn @"testErrorReturn
def testErrorReturn() raises:
    var input: String
    # CHECK: try
    with MyStringReturningCtx() as ctx:
        # CHECK-NOT: @MyStringReturningCtx::@"__deinit__
        var x = (
            ctx.read()
        )  # expected-warning {{assignment to 'x' was never used}}
        input = "hello"
    # CHECK: except
    use(input)


# COM: Test partial destruction of initialized fields upon an error return.
struct Field(ImplicitlyCopyable):
    def __init__(out self, *, copy: Self):
        pass

    def __deinit__(deinit self):
        pass


# CHECK-LABEL: lit.struct.decl @DestructSome
struct DestructSome(Movable where False):
    var a: Field
    var b: Field

    # CHECK-LABEL: lit.fn @"__init__
    def __init__(out self, a: Field, b: Field) raises:
        # CHECK:      lifetime.start %__call_result_tmp__
        # CHECK-NEXT: call {{.*}}somethingThatRaises
        # CHECK-NEXT: if
        # CHECK-NEXT:   mark_consumed
        # CHECK-NEXT:   lifetime.end %__call_result_tmp__
        # CHECK-NEXT:   kgen.param.constant
        # CHECK-NEXT:   lit.error_return
        somethingThatRaises()

        # CHECK: [[FIELD:%.*]] = lit.ref.struct.ger %self[a]
        # CHECK-NEXT: __init__{{.*}}"{{.*}}(%a, [[FIELD]]){{.*}}*, "copy"
        self.a = a

        # CHECK:      lifetime.start %__call_result_tmp__
        # CHECK-NEXT: call {{.*}}somethingThatRaises
        # CHECK-NEXT: if
        # CHECK-NEXT:   [[FIELD:%.*]] = lit.ref.struct.ger %self[a]
        # CHECK-NEXT:   __deinit__{{.*}}([[FIELD]])
        # CHECK-NEXT:   mark_consumed %__call_result_tmp__
        # CHECK-NEXT:   lifetime.end %__call_result_tmp__
        # CHECK-NEXT:   kgen.param.constant
        # CHECK-NEXT:   lit.error_return
        somethingThatRaises()

        # CHECK: [[FIELD:%.*]] = lit.ref.struct.ger %self[b]
        # CHECK-NEXT: __init__{{.*}}"{{.*}}(%b, [[FIELD]]){{.*}}*, "copy"
        self.b = b

        # At this point 'self' is fully initialized, so any exit out should
        # destroy the whole thing.

        # CHECK:      lifetime.start %__call_result_tmp__
        # CHECK-NEXT: call {{.*}}somethingThatRaises
        # CHECK-NEXT: if
        # CHECK-NEXT:   __deinit__{{.*}}(%self)
        # CHECK-NEXT:   mark_consumed %__call_result_tmp__
        # CHECK-NEXT:   lifetime.end %__call_result_tmp__
        # CHECK-NEXT:   kgen.param.constant
        # CHECK-NEXT:   lit.error_return
        somethingThatRaises()


def borrow_and_return(value: MemExample) raises -> MemExample:
    return value


def use(err: Error):
    pass


# CHECK-LABEL: lit.fn @"raising_use
def raising_use(var value: MemExample):
    try:
        # CHECK:      [[BORROW:%.*]] = lit.ref.immut %value
        # CHECK-NEXT: [[VAL:%.*]] = lit.var.decl "__call_result_tmp__"
        # CHECK-NEXT: lifetime.start %__try_error__
        # CHECK-NEXT: lifetime.start [[VAL]]
        # CHECK-NEXT: [[IS_ERR:%.*]] = lit.call {{.*}}borrow_and_return{{.*}}([[BORROW]], %__try_error__, [[VAL]])
        # CHECK-NEXT: call {{.*}}@MemExample::@"__deinit__{{.*}}(%value)
        # CHECK-NEXT: if [[IS_ERR]]
        # CHECK-NEXT:   call {{.*}}@Error::@"__deinit__{{.*}}(%__try_error__)
        # CHECK-NEXT:   lit.var.lifetime.end %__try_error__
        # CHECK-NEXT:   mark_consumed [[VAL]]
        # CHECK-NEXT:   lit.var.lifetime.end [[VAL]]
        # CHECK-NEXT:   lit.try.raise
        # CHECK-NEXT: } else {
        # CHECK-NEXT:   mark_consumed %__try_error__
        # CHECK-NEXT:   lit.var.lifetime.end %__try_error__
        # CHECK-NEXT:   hlcf.yield
        # CHECK-NEXT: }
        # CHECK-NEXT: lit.ownership.use [[VAL]]
        # CHECK-NEXT: call {{.*}}@MemExample::@"__deinit__{{.*}}([[VAL]])
        # CHECK-NEXT: lifetime.end [[VAL]]
        _ = borrow_and_return(value)
    except:
        pass


# CHECK-LABEL: lit.struct.decl @ThrowingSelfInit
struct ThrowingSelfInit(Movable where False):
    var x: Int

    def __deinit__(deinit self):
        pass

    # CHECK-LABEL: lit.fn @"__init__
    def __init__(out self) raises:
        self.x = 0

    # CHECK-LABEL: lit.fn @"__init__
    def __init__(out self, x: Int) raises:
        # CHECK-NEXT: [[IS_ERR:%.*]] = lit.call {{.*}}__init__{{.*}}(%__error__, %self)
        # CHECK-NEXT: if [[IS_ERR]]
        # CHECK-NEXT:   mark_consumed %self
        # CHECK-NEXT:   [[TRUE:%.*]] = kgen.param.constant
        # CHECK-NEXT:   error_return [[TRUE]]
        # CHECK-NEXT: else
        # CHECK-NEXT:   mark_consumed %__error__
        # CHECK-NEXT:   yield
        self = ThrowingSelfInit()

    # CHECK-LABEL: lit.fn @"__init__
    def __init__(out self, x: Int, y: Int) raises:
        # CHECK-NEXT: [[IS_ERR:%.*]] = lit.call {{.*}}__init__{{.*}}(%__error__, %self)
        # CHECK:      else
        # CHECK-NEXT:   call {{.*}}__deinit__{{.*}}(%self)
        # CHECK-NEXT:   mark_consumed %__error__
        # CHECK-NEXT:   yield
        self = ThrowingSelfInit()
        # CHECK:      lit.call {{.*}}__init__{{.*}}(%__error__, %self)
        self = ThrowingSelfInit()


struct InitFieldsDestroyedInThrowingConstructor(Movable where False):
    var x: MemExample

    def __init__(out self):
        self.x = MemExample()

    # CHECK-LABEL: lit.fn @"__init__(!kgen.scalar<bool>)"
    def __init__(out self, cond: __mlir_type.`!kgen.scalar<bool>`) raises:
        self = InitFieldsDestroyedInThrowingConstructor()
        # CHECK:      hlcf.elif %cond {
        # CHECK-NEXT:   lit.call {{.*}}__deinit__{{.*}}(%self)
        # CHECK-NEXT:   lit.call {{.*}}::@Error::@"__init__
        # CHECK-NEXT:   kgen.param.constant
        # CHECK-NEXT:   lit.error_return
        # CHECK-NEXT: } else {
        # CHECK-NEXT:   hlcf.yield
        # CHECK-NEXT: }
        if cond:
            raise Error()


# CHECK-LABEL: lit.fn @"doesnt_actually_raise{{.*}}%__error__: !lit.ref<:non_struct_type #alias_Never
def doesnt_actually_raise() raises Never:
    pass


# CHECK-LABEL: lit.fn @"test_doesnt_actually_raise{{.*}}() -> !alias_Int1
# CHECK-NEXT: %__never_error__ = lit.var.decl
# CHECK-NEXT: %__call_result_tmp__ = lit.var.decl
# CHECK-NEXT: lit.var.lifetime.start %__never_error__
# CHECK-NEXT: lit.var.lifetime.start %__call_result_tmp__
# CHECK-NEXT: lit.call {{.*}}doesnt_actually_raise{{.*}}(%__never_error__, %__call_result_tmp__)
# CHECK-NEXT: lit.var.lifetime.end %__call_result_tmp__
# CHECK-NEXT: lit.var.lifetime.end %__never_error__
# CHECK-NEXT: kgen.param.constant: !alias_Int1 = <rebind(:!Int {:scalar<index> 42})>
def test_doesnt_actually_raise() -> Int:
    doesnt_actually_raise()
    return 42
