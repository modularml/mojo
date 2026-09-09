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
# Raise and Try
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"simpleTryExcept
def simpleTryExcept():
    var a: Int
    # CHECK: lit.try
    try:
        # CHECK: lit.ref.store
        a = 0
        # CHECK-NEXT: lit.try.yield
    # CHECK-NEXT: except
    except:
        # CHECK-NEXT: lit.try.yield
        pass
    # CHECK-NEXT: else
    # CHECK-NEXT: lit.try.yield
    # CHECK: lit.end_fn


# CHECK-LABEL: lit.fn @"tryExceptElse
def tryExceptElse():
    var a: Int
    # CHECK: lit.try
    try:
        pass
    except:
        pass
    # CHECK: else
    else:
        # CHECK: lit.ref.store
        a = 0
        # CHECK-NEXT: lit.try.yield


def eatError(err: Error):
    pass


# CHECK-LABEL: lit.fn @"tryExceptArgDef
def tryExceptArgDef():
    try:
        pass
    # CHECK: } except {
    except err:
        # CHECK-NEXT: [[ERR:%.*]] = lit.ref.immut %err
        # CHECK-NEXT: eatError{{.*}}([[ERR]])
        eatError(err)


# CHECK-LABEL: lit.fn @"tryFinally
def tryFinally():
    # CHECK: lit.try
    try:
        # CHECK-NEXT: lit.try.yield
        pass
    # CHECK-NEXT: except
    # CHECK-NEXT: unreachable
    # CHECK: finally
    finally:
        # CHECK: lit.return
        return
    # CHECK: lit.try %__try_error___0
    try:
        # CHECK: lit.try
        try:
            # CHECK-NEXT: lit.try.yield
            pass
        # CHECK-NEXT: except
        # CHECK-NEXT: lit.call {{.*}}Error::@"__init__{{.*}}"{{.*}}(%__try_error___1, %__try_error___0){{.*}}*, "move"
        # CHECK-NEXT: lit.raise
        finally:
            pass
    except:
        pass


def maybeRaises() raises -> Int:
    return 0


# CHECK-LABEL: lit.fn @"propagateErrorInDef
def propagateErrorInDef() raises:
    # CHECK: %a = lit.var.decl "a"
    # CHECK: lit.call {{.*}}maybeRaises{{.*}}(%__error__, %a)
    var a = maybeRaises()


# CHECK-LABEL: lit.fn @"propagateErrorInRaisingFn
def propagateErrorInRaisingFn() raises:
    # CHECK:  %a = lit.var.decl {{.*}} : !lit.ref<:meta<!Int> #alias_Int,
    var a: Int
    # CHECK:  lit.call {{.*}}maybeRaises{{.*}}(%__error__, %a)
    a = maybeRaises()


# CHECK-LABEL: lit.fn @"propagateErrorInTry
def propagateErrorInTry():
    var a: Int
    # CHECK: lit.try
    try:
        # CHECK-NEXT: lit.call {{.*}}maybeRaises{{.*}}(%__try_error__, %a)
        a = maybeRaises()
        # CHECK-NEXT: lit.try.yield
    except:
        pass


# CHECK-LABEL: lit.fn @"raiseError
def raiseErrorInDef() raises:
    # CHECK: lit.call {{.*}}@Error::@"__init__{{.*}}(%__error__)
    # CHECK-NEXT: lit.raise
    raise Error()


# CHECK-LABEL: lit.fn @"raiseErrorInIf
def raiseErrorInIf(cond: Bool) raises:
    # CHECK: hlcf.elif
    if cond:
        # CHECK: lit.call {{.*}}@Error::@"__init__{{.*}}(%__error__)
        # CHECK-NEXT: lit.raise
        raise Error()


# CHECK-LABEL: lit.fn @"raiseErrorInTry
def raiseErrorInTry():
    # CHECK: lit.try %__try_error__
    try:
        # CHECK-NEXT: lit.call {{.*}}@Error::@"__init__{{.*}}(%__try_error__)
        # CHECK-NEXT: lit.raise
        raise Error()
    except:
        pass


# CHECK-LABEL: lit.fn @"rethrowsToRethrow
def rethrowsToRethrow():
    # CHECK: lit.try [[TRY_ERROR1:%.*]] :
    try:
        # CHECK: lit.try [[TRY_ERROR2:%.*]] :
        try:
            # CHECK: lit.call {{.*}}maybeRaises{{.*}}([[TRY_ERROR2]], %__call_result_tmp__
            maybeRaises()  # expected-warning {{'Int' value is unused; assign to '_' to discard the result}}
        # CHECK: } except {
        except:
            # CHECK-NEXT: lit.call {{.*}}Error::@"__init__{{.*}}"{{.*}}([[TRY_ERROR2]], [[TRY_ERROR1]]){{.*}}*, "move"
            # CHECK-NEXT: lit.raise
            raise
        # CHECK: }
    # CHECK: } except {
    except:
        # CHECK: lit.return %none
        return


struct S(ImplicitlyCopyable, Movable):
    var v: Int

    @implicit
    def __init__(out self, x: Int):
        self.v = x

    def __init__(out self) raises:
        self.v = 1


def fail() raises -> S:
    return 0


# CHECK-LABEL: lit.fn @"call_raising
def call_raising():
    # CHECK: lit.try %e
    try:
        # CHECK: [[ERR:%.*]] =  lit.call {{.*}}::@"fail{{.*}}(%e, %x)
        var x = fail()
        # CHECK: %y = lit.var.decl "y"
        # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%e, %y)
        var y = S()
        # CHECK-NEXT: lit.try.yield
    except e:
        pass


def fail_raises() raises -> S:
    return fail()


def fail_register() raises -> Int:
    return 0


# CHECK-LABEL: lit.fn @"fail_register_raises
def fail_register_raises() raises -> Int:
    # CHECK-NEXT: call {{.*}}fail_register{{.*}}(%__error__, %__result__)
    # CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: scalar<bool> = <false>
    # CHECK-NEXT: lit.return [[FALSE]]
    return fail_register()
