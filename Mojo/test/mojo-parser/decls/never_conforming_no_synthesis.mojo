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

# Synthesis-suppression matrix for never-conforming traits. A conformance whose
# condition can never hold -- whether spelled literally `where False`,
# contradicted by the struct's own arithmetic where-clause, contradicted via a
# `conforms_to` negation, or living in an unsatisfiable (`where False`) ambient
# scope -- is an opt-out: no special member (move ctor, copy ctor,
# default-trait-method wrapper) is synthesized, and no "does not implement all
# requirements" error fires. Every spelling in a trait's row reaches the same
# verdict; that equality is the invariant under test (MOCO-4135).
#
# The use-site "value has no attribute" diagnostics for these opt-outs are
# pinned separately in `traits_errors.mojo`.
#
# The ambient-false spelling (`... where False` on the struct) additionally makes
# the struct linear: an unsatisfiable body clause contradicts even the
# unconditionally-injected `Deinitable` slot, so the struct is
# uninstantiable anyway. The shared invariant asserted per row -- no move/copy/
# defaulted-method synthesis -- still holds across all four spellings.

# RUN: %parse-mojo-isolated %s | FileCheck %s


trait Greeter:
    def greet(self, name: String) -> String:
        return "Hello, " + name


# --- Movable: no move constructor synthesized -------------------------------

# CHECK-LABEL: lit.struct.decl @MovableLiteral
# CHECK-NOT: __init__(move:
struct MovableLiteral(Movable where False):
    pass

# CHECK-LABEL: lit.struct.decl @MovableArith
# CHECK-NOT: __init__(move:
struct MovableArith[n: Int](Movable where not (n > 0)) where n > 0:
    pass

# CHECK-LABEL: lit.struct.decl @MovableConfNeg
# CHECK-NOT: __init__(move:
struct MovableConfNeg[T: AnyType](
    Movable where conforms_to(T, Movable)
) where not conforms_to(T, Movable):
    pass

# CHECK-LABEL: lit.struct.decl @MovableAmbient
# CHECK-NOT: __init__(move:
struct MovableAmbient[T: AnyType](
    Movable where conforms_to(T, Movable)
) where False:
    pass


# --- Copyable: no copy constructor synthesized ------------------------------

# CHECK-LABEL: lit.struct.decl @CopyableLiteral
# CHECK-NOT: )Copyable
struct CopyableLiteral(Copyable where False):
    pass

# CHECK-LABEL: lit.struct.decl @CopyableArith
# CHECK-NOT: )Copyable
struct CopyableArith[n: Int](Copyable where not (n > 0)) where n > 0:
    pass

# CHECK-LABEL: lit.struct.decl @CopyableConfNeg
# CHECK-NOT: )Copyable
struct CopyableConfNeg[T: AnyType](
    Copyable where conforms_to(T, Copyable)
) where not conforms_to(T, Copyable):
    pass

# CHECK-LABEL: lit.struct.decl @CopyableAmbient
# CHECK-NOT: )Copyable
struct CopyableAmbient[T: AnyType](
    Copyable where conforms_to(T, Copyable)
) where False:
    pass


# --- User trait with a defaulted method: no wrapper synthesized -------------

# CHECK-LABEL: lit.struct.decl @GreeterLiteral
# CHECK-NOT: greet(
struct GreeterLiteral(Greeter where False):
    pass

# CHECK-LABEL: lit.struct.decl @GreeterArith
# CHECK-NOT: greet(
struct GreeterArith[n: Int](Greeter where not (n > 0)) where n > 0:
    pass

# CHECK-LABEL: lit.struct.decl @GreeterConfNeg
# CHECK-NOT: greet(
struct GreeterConfNeg[T: AnyType](
    Greeter where conforms_to(T, Greeter)
) where not conforms_to(T, Greeter):
    pass

# CHECK-LABEL: lit.struct.decl @GreeterAmbient
# CHECK-NOT: greet(
struct GreeterAmbient[T: AnyType](
    Greeter where conforms_to(T, Greeter)
) where False:
    pass


# --- Deinitable: struct becomes linear, same message per spelling --

# CHECK-LABEL: lit.struct.decl @IDLiteral
# CHECK-SAME: does not conform to 'Deinitable' and must be explicitly destroyed
struct IDLiteral(Deinitable where False):
    pass

# CHECK-LABEL: lit.struct.decl @IDArith
# CHECK-SAME: does not conform to 'Deinitable' and must be explicitly destroyed
struct IDArith[n: Int](Deinitable where not (n > 0)) where n > 0:
    pass


# Sentinel: bounds the final `CHECK-NOT`/`CHECK-SAME` region before the stdlib
# decls (which legitimately synthesize move/copy ctors) appear in the dump.
# CHECK-LABEL: lit.struct.decl @MatrixSentinel
struct MatrixSentinel:
    pass
