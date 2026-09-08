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
#
# Tests for interior origins.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt -lower-semantic-cf -check-lifetimes -verify-parameters -verify-diagnostics

def use_any[*Ts: AnyType](*args: *Ts): pass

# ===----------------------------------------------------------------------=== #
# Test Type
# ===----------------------------------------------------------------------=== #

struct MyList[T: AnyType](Movable where False):
    var data: UnsafePointer[Self.T, UntrackedOrigin[mut=True]]

    def __init__(out self):
        self.data = UnsafePointer[Self.T, UntrackedOrigin[mut=True]].unsafe_dangling()

    def __deinit__(deinit self):
        pass # Explicit deinit so it isn't considered trivial and elided.

    def mutate(mut self): # must invalidate all interior origins.
        pass

    def access(self): # doesn't invalidate interior origins.
        pass

    @__unsafe_nested_origins_read_only
    def __getitem__(
        ref self
    ) -> ref[self.data._get_ref_with_unsafe_interior_origin["element"](self)] Self.T:
        return self.data._get_ref_with_unsafe_interior_origin["element"](self)

# ===----------------------------------------------------------------------=== #
# Basic Tests
# ===----------------------------------------------------------------------=== #

def test_invalidate_base():
    # expected-note @+1 {{'list' declared here}}
    var list = MyList[Int]()

    ref elt_ref1 = list[]
    elt_ref1 += 4
    list^.__deinit__()

    # Deleting list obviously invalidates it.
    # expected-error @+1 {{use of uninitialized value 'list'}}
    elt_ref1 += 4

def test_invalidate_interior():
    var list = MyList[Int]()
    ref elt_ref2 = list[]
    elt_ref2 += 4
    list.mutate()   # expected-note {{origin was invalidated here}}
    # expected-error @+1 {{use of invalidated interior reference 'list["element"]'}}
    elt_ref2 += 4

# simple control flow test.
def test_if(cond: Bool):
    var list = MyList[Int]()
    ref elt_ref2 = list[]
    elt_ref2 += 4
    if cond:
        list.mutate()   # expected-note {{origin was invalidated here}}
    # expected-error @+1 {{use of invalidated interior reference 'list["element"]'}}
    elt_ref2 += 4

struct TwoIntLists(Movable where False):
   var first: MyList[Int]
   var second: MyList[Int]

   def __init__(out self):
      self.first = MyList[Int]()
      self.second = MyList[Int]()

# Test that we can handle nested field sensitivity correctly.
def test_field_sensitive_nested_invalidation():
    var list_of_two_intlists = MyList[TwoIntLists]()
    ref first_list = list_of_two_intlists[].first
    ref second_list = list_of_two_intlists[].second

    ref first_list_elt = first_list[]
    ref second_list_elt = second_list[]

    # Mutating the elements of either list is fine, and shouldn't cause a
    # problem for anything.
    first_list_elt += 4
    second_list_elt += 4

    # Mutating the first list shouldn't invalidate the second list because of
    # nested field sensitivity.
    first_list.mutate()   # expected-note {{origin was invalidated here}}
    second_list_elt += 4

    # However, it should invalidate the first list.
    # expected-error @+1 {{use of invalidated interior reference 'list_of_two_intlists["element"].first["element"]'}}
    first_list_elt += 4

def test_dominance_lifetime():
    var list = MyList[Int]()
    ref elt_ref2 = list[]
    elt_ref2 += 4 # This works
    list.mutate()

    ref elt_ref3 = list[]  # expected-note {{origin was defined here, after the reference was formed}}
    elt_ref3 += 4   # This works because the ref was refreshed.

    # This is an error because the ref is invalidated.
    # expected-error @+1 {{use of invalidated interior reference 'list["element"]'}}
    elt_ref2 += 4

def test_dominance_lifetime2(cond: Bool):
    var list = MyList[Int]()
    ref elt_ref = list[]
    elt_ref += 4 # This works

    # This is all fine, nothing invalidates elt_ref.
    var sum = 0
    if cond:
        sum += elt_ref
        list.access() # doesn't invalidate anything.
    else:
        sum += elt_ref
    sum += elt_ref

    # This is not fine, because the list is mutated, potentially invalidating
    # elt_ref.  It gets reinitialized within the body, but that doesn't make
    # elt_ref valid again!
    if cond: # expected-note {{origin was defined inside this control structure}}
        list.mutate()
        ref elt_ref_a = list[]
        elt_ref_a += 4   # This works because the ref was refreshed.
    else:
        list.mutate()
        ref elt_ref_b = list[]
        elt_ref_b += 4   # This works because the ref was refreshed.

    # This is an error because the ref is invalidated.
    # expected-error @+1 {{use of invalidated interior reference 'list["element"]'}}
    elt_ref += 4

# This verifies we're handling hlcf.elif dominance correctly.
def test_rederive_in_controlled_block(list: MyList[Int]) -> Int:
    if list[] > 0:
        return list[]
    return 0

def test_rederive_after_mutation_in_block(cond: Bool, mut list: MyList[Int]):
    ref r = list[]
    if cond:
        list.mutate() # expected-note {{origin was invalidated here}}
        # expected-error @+1 {{use of invalidated interior reference}}
        r += 1

# MOCO-4344 - Variadic packs + interior origins.
def test_rederive_interior_in_variadic_pack(list: MyList[Int]):
    var n = 0
    ref k = n
    use_any(list[])
    use_any(k, list[])

# MOCO-4345 - vardecl store handling
def test_vardecl_store_handling(list: MyList[Int]):
    var p: type_of(Pointer(to=list[]))
    p = Pointer(to=list[])
    _ = p

def test_bad_io_concrete[attr: StringLiteral](ptr: UnsafePointer[Int, _]) -> Int:
    var str = String()
    # expected-error @+1 {{interior origin name must be a string literal, it cannot be parametric when used}}
    ref r = ptr._get_ref_with_unsafe_interior_origin[attr](str)
    return r

def ret_invalid_ref(mut list: MyList[Int]) -> ref[list[]] Int:
    ref r = list[]
    list.mutate()  # expected-note {{origin was invalidated here}}
    return r # expected-error {{use of invalidated interior reference 'list["element"]'}}

# This shouldn't crash.
def test_thing1(data: MyList[String]) raises -> ref[data[]] String:
    return data[]
def test_thing2(data: MyList[String]) raises -> ref[data[]] String:
    return test_thing1(data)

# This shouldn't crash.
def throwing_function(ref values: MyList[String]) raises -> ref[values[]] String:
    return values[]

def call_throwing_fn(list: MyList[String]) raises:
    _ = throwing_function(list)

struct StructWithList(Movable where False):
    var data: MyList[String]

    def add_type(mut self) -> ref[self.data[]] String:
        return self.data[]

def test_struct_with_list(var thing: StructWithList):
    _ = thing.add_type()


struct KeyErr(ErrorConversionTrait):
    pass
def access_or_raise_custom_error(a: MyList[Int])raises KeyErr -> ref[a[]] Int :
    return a[]

# MOCO-4371
def test_access_or_raise_custom_error_convert_to_error() raises:
    var l = MyList[Int]()
    # This should be valid.
    var x = access_or_raise_custom_error(l)
    _ = x

# MOCO-4374
def test_problem_4374():
    var lst = MyList[Int]()
    var p = Pointer(to=lst[])
    _ = p[]

# ===----------------------------------------------------------------------=== #
# Closures
# ===----------------------------------------------------------------------=== #

def test_nested_fn1():
    var base = MyList[Int]()
    var p = Pointer(to=base[])

    @__copy_capture(p)
    @__parameter
    def inner():
        _ = p == p  # Should be fine.
    inner()

def test_nested_fn2():
    var base = MyList[Int]()
    var p = Pointer(to=base[])

    @__copy_capture(p)
    @__parameter
    def inner():
        _ = p == p  # Should be fine.
        # expected-error @+1 {{incorrect invalidation of interior origin in closure 'base["element"]'}}
        base.mutate()
    inner()

def test_nested_fn3():
    var base = MyList[Int]()
    var p = Pointer(to=base[])

    @__copy_capture(p)
    @__parameter
    def inner():
        _ = p == p  # Should be fine.

    base.mutate() # expected-note {{origin was invalidated here}}
    inner() # expected-error {{use of invalidated interior reference 'base["element"]'}}

def test_nested_fn4():
    var base = MyList[Int]()
    var p = Pointer(to=base[])

    @__copy_capture(p)
    @__parameter
    def inner():
        _ = p == p  # Should be fine.

    base.mutate()
    _ = base[] # attempt to refresh.
    # FIXME: This should be an error; we should make sure we have the same
    # version of the reference as when the closure was created.
    inner()

def test_loop_iteration(var list: List[Int]):
    # The mutate method should invalidate the active iterator.
    # expected-error @+1 {{use of invalidated interior reference 'list["element"]'}}
    for _ in list:
        # expected-note @+1 {{origin was invalidated here}}
        list.append(4)
