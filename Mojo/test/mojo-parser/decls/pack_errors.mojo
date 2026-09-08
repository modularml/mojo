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

# RUN: %parse-mojo-isolated -verify-diagnostics %s

##===----------------------------------------------------------------------===##
# Unpacking
##===----------------------------------------------------------------------===##


# expected-note @+1 {{function declared here}}
def test_unpack(d: Int):
    # expected-error @+1 {{kwargs unpack can only satisfy another kwargs argument, it can't fulfill 'd'}}
    test_unpack(**d)
    # expected-error @+1 {{unpack is only supported when the callee accepts a variadic; to forward a runtime pack to a fixed-arity callee, route the call through a dispatcher whose argument is itself a variadic pack (e.g. `def shim[Ts: TypeList[Trait=AnyType, ...], //, callee: def(*args: *Ts) thin](...): callee(*pack)`)}}
    test_unpack(*d)
    # expected-error @+2 {{positional argument must not follow an unpack; move it before or convert to a keyword argument}}
    # expected-note @+1 {{unpacked positional argument specified here}}
    test_unpack(*d, d)


def test_unpack_twice(d: Int):
    # expected-error @+2 {{unpack markers (*name syntax) must not appear more than once in a call; remove the second unpack}}
    # expected-note @+1 {{previous unpacked positional argument specified here}}
    test_unpack_twice(*d, *d)


# `f(*args, **kwargs)` is valid; the reverse order is not, matching Python.
def test_unpack_order(*args: Int, var **kwargs: Int):
    # expected-error @+1 {{positional unpack must not follow keyword unpacking; move it before the '**' unpack}}
    test_unpack_order(**kwargs, *args)


# expected-note @+1 {{function declared here}}
def test_merge_kwargs(var **kwargs: Int):
    # expected-error @+2 {{combining a '**' unpack with other keyword arguments for '**kwargs' is not supported}}
    # expected-note @+1 {{other keyword argument specified here}}
    test_merge_kwargs(a=1, **kwargs^)


# expected-note @+1 {{function declared here}}
def test_two_kwargs_unpacks(var **kwargs: Int):
    # expected-error @+2 {{combining a '**' unpack with other keyword arguments for '**kwargs' is not supported}}
    # expected-note @+1 {{other keyword argument specified here}}
    test_two_kwargs_unpacks(**kwargs^, **kwargs^)


# expected-note @below {{function declared here}}
def test_sum_intable[*Ts: Intable](*pack: *Ts):
    pass


def test_forward_any_pack[*Ts: AnyType](*pack: *Ts):
    # expected-error @+1 {{invalid call to 'test_sum_intable': cannot unpack a pack of type 'AnyType' ('Ts.values') into a call that expects a pack of type 'Intable' ('Ts.values')}}
    test_sum_intable(*pack)


# Splatting a pack into a fixed-arity callee is not supported. The
# count-mismatch diagnostic should attach a hint pointing at the
# variadic-pack dispatcher pattern.
# expected-note @+1 {{function declared here}}
def takes_two_ints(x: Int, y: Int):
    pass


def test_splat_into_fixed_arity[*Ts: AnyType](*pack: *Ts):
    # expected-error @+1 {{unpack is only supported when the callee accepts a variadic; to forward a runtime pack to a fixed-arity callee, route the call through a dispatcher whose argument is itself a variadic pack (e.g. `def shim[Ts: TypeList[Trait=AnyType, ...], //, callee: def(*args: *Ts) thin](...): callee(*pack)`)}}
    takes_two_ints(*pack)


# Resolved-pack splats: even when the operand's `VariadicPack` element list
# is statically known (here, by spelling concrete element types into the
# pack-typed parameter), splatting a pack into a fixed-arity callee is not
# supported. As with the symbolic case above, the call is diagnosed with
# the dispatcher-pattern hint regardless of whether the resolved element
# count is too few, too many, or an exact match for the callee's arity;
# `*pack`-to-positional expansion is follow-up work (MOCO-2210).
# expected-note @+1 {{function declared here}}
def takes_three_ints(a: Int, b: Int, c: Int):
    pass


def test_resolved_pack_too_few(
    pack: VariadicPack[origin=_, element_trait=AnyType, _, Int, Int]
):
    # expected-error @+1 {{unpack is only supported when the callee accepts a variadic; to forward a runtime pack to a fixed-arity callee, route the call through a dispatcher whose argument is itself a variadic pack (e.g. `def shim[Ts: TypeList[Trait=AnyType, ...], //, callee: def(*args: *Ts) thin](...): callee(*pack)`)}}
    takes_three_ints(*pack)


# expected-note @+1 {{function declared here}}
def takes_one_int(a: Int):
    pass


def test_resolved_pack_too_many(
    pack: VariadicPack[origin=_, element_trait=AnyType, _, Int, Int]
):
    # expected-error @+1 {{unpack is only supported when the callee accepts a variadic; to forward a runtime pack to a fixed-arity callee, route the call through a dispatcher whose argument is itself a variadic pack (e.g. `def shim[Ts: TypeList[Trait=AnyType, ...], //, callee: def(*args: *Ts) thin](...): callee(*pack)`)}}
    takes_one_int(*pack)


# expected-note @+1 {{function declared here}}
def takes_int_and_float(x: Int, y: Float64):
    pass


def test_resolved_pack_matched_arity(
    pack: VariadicPack[origin=_, element_trait=AnyType, _, Int, Float64]
):
    # expected-error @+1 {{unpack is only supported when the callee accepts a variadic; to forward a runtime pack to a fixed-arity callee, route the call through a dispatcher whose argument is itself a variadic pack (e.g. `def shim[Ts: TypeList[Trait=AnyType, ...], //, callee: def(*args: *Ts) thin](...): callee(*pack)`)}}
    takes_int_and_float(*pack)


# Unpacking into a *args (pos-vararg) function instead of a typed pack.
# Users may confuse *args-style variadics with typed packs.
# expected-note @+1 {{function declared here}}
def takes_varargs(*args: Int):
    pass


def test_unpack_into_varargs[*Ts: AnyType](*pack: *Ts):
    # expected-error @+1 {{value passed to 'args' cannot be converted from 'VariadicPack[False, Ts]' to 'VariadicList[Int, False]'}}
    takes_varargs(*pack)


def test_unpack_varargs(*args: Int):
    # expected-error @+1 {{unpack is only supported when the callee accepts a variadic; to forward a runtime pack to a fixed-arity callee, route the call through a dispatcher whose argument is itself a variadic pack (e.g. `def shim[Ts: TypeList[Trait=AnyType, ...], //, callee: def(*args: *Ts) thin](...): callee(*pack)`)}}
    test_unpack(*args)


# Double-star unpacking a real **kwargs pack.
def takes_kwargs(var **kwargs: Int):
    pass


def test_unpack_kwargs_pack(var **kwargs: Int):
    # expected-error @below {{value of type 'StringDict[Int]' cannot be implicitly copied}}
    # expected-note @below {{consider transferring the value with '^'}}
    takes_kwargs(**kwargs)
    # Ok!
    takes_kwargs(**kwargs^)


# expected-note @+1 {{function declared here}}
def takes_varpack[*Ts: AnyType](*pack: *Ts):
    pass


def test_positional_concat[*Ts: AnyType](*pack: *Ts):
    # expected-error @below {{concatenating unpacked positional arguments is not supported}}
    takes_varpack("hello", *pack)

    # expected-error @below {{positional argument must not follow an unpack; move it before or convert to a keyword argument}}
    # expected-note @below {{unpacked positional argument specified here}}
    takes_varpack(*pack, "hello")


def test_unpack_twice[*Ts: AnyType](*pack: *Ts):
    # expected-error @below {{unpack markers (*name syntax) must not appear more than once in a call; remove the second unpack}}
    # expected-note @below {{previous unpacked positional argument specified here}}
    takes_varpack(*pack, *pack)


# Splatting a non-VariadicPack type is not allowed.
struct MyStruct(Movable where False):
    var x: Int

    def __init__(out self, x: Int):
        self.x = x


# expected-note @+1 {{function declared here}}
def takes_variadic_pack2[*Ts: AnyType](*pack: *Ts):
    pass


def test_splat_non_pack():
    var s = MyStruct(42)
    # expected-error @+1 {{cannot unpack value of type 'MyStruct' into a variadic pack argument; expected a VariadicPack}}
    takes_variadic_pack2(*s)
