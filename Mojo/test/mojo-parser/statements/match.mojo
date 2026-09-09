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

# RUN: %parse-mojo-isolated %s | FileCheck %s

# A function we can call with minimal IR gruff but still verify the right
# code is put out in the right place.
def case_callee[p: Int](): pass

# CHECK-LABEL: lit.fn @"match_bindings
def match_bindings(i: Int, var s: String):
    # Bare bindings are "imm" bindings: register values become `bound`,
    # memory values become an immutable `ref` (muttoimm).
    # CHECK:       [[BX:%.*]] = lit.var.decl "x" bound
    # CHECK:       lit.ref.store %i, [[BX]]
    # CHECK:       lit.call {{.*}}@"__add__(
    __match i:
    case x:
        _ = x+1

    # CHECK:       [[SX:%.*]] = lit.var.decl "x" ref
    # CHECK:       lit.ref.store {{.*}}, [[SX]]
    # CHECK:       lit.ref.load [[SX]]
    # CHECK:       lit.call {{.*}}@"__len__(::String)"{{.*}}muttoimm
    __match s:
    case x:
        _ = x.__len__()

    # 'var' bindings make a copy.
    # CHECK:       [[VX:%.*]] = lit.var.decl "x" var
    # CHECK:       lit.ref.store %i, [[VX]]
    # CHECK:       lit.call {{.*}}@"__iadd__({{.*}}([[VX]],
    __match i:
    case var x:
        x += 1

    # CHECK:       [[VS:%.*]] = lit.var.decl "x" var
    # CHECK:       lit.call {{.*}}@"__init__(copy:::String)"{{.*}}[[VS]]
    # CHECK:       lit.call {{.*}}@"__iadd__{{.*}}([[VS]],
    __match s:
    case var x:
        x += "x"

    # Ref bindings work with mutable references to the original.
    # CHECK:       [[RX:%.*]] = lit.var.decl "x" ref
    # CHECK:       lit.ref.store %s, [[RX]]
    # CHECK:       lit.call {{.*}}@"__iadd__{{.*}}[mut *"s`
    __match s:
    case ref x:
        x += "x"


# CHECK-LABEL: lit.fn @"match_same_indent
# CHECK-NEXT:    [[L0:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 0}>
# CHECK-NEXT:    [[EQ0:%.*]] = lit.call {{.*}}@"__eq__({{.*}}(%x, [[L0]])
# CHECK-NEXT:    [[B0:%.*]] = lit.call {{.*}}@"__mlir_bool__(::Bool)"([[EQ0]])
# CHECK-NEXT:    hlcf.elif [[B0]] {
# CHECK-NEXT:      lit.call {{.*}}@"case_callee{{.*}}<index> 0
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    } else {
# CHECK-NEXT:      [[L1:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 1}>
# CHECK-NEXT:      [[EQ1:%.*]] = lit.call {{.*}}@"__eq__({{.*}}(%x, [[L1]])
# CHECK-NEXT:      [[B1:%.*]] = lit.call {{.*}}@"__mlir_bool__(::Bool)"([[EQ1]])
# CHECK-NEXT:      hlcf.elif.yield [[B1]]
# CHECK-NEXT:    } then {
# CHECK-NEXT:      lit.call {{.*}}@"case_callee{{.*}}<index> 1
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    } else {
# CHECK-NEXT:      [[T2:%.*]] = kgen.param.constant: scalar<bool> = <true>
# CHECK-NEXT:      hlcf.elif.yield [[T2]]
# CHECK-NEXT:    } then {
# CHECK-NEXT:      lit.call {{.*}}@"case_callee{{.*}}<index> 2
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    } else {
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    }
def match_same_indent(x: Int):
    __match x:
    case 0:
        case_callee[0]()
    case 1:
        case_callee[1]()
    case _:
        case_callee[2]()


# CHECK-LABEL: lit.fn @"match_indented_cases
# CHECK:         kgen.param.constant: !Int = <{:scalar<index> 0}>
# CHECK:         lit.call {{.*}}@"__eq__(
# CHECK:         lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:       hlcf.elif %{{.*}} {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 0
# CHECK:         hlcf.yield
# CHECK:       } else {
def match_indented_cases(x: Int):
    __match x:
        case 0:
            case_callee[0]()
        case _:
            case_callee[1]()


# CHECK-LABEL: lit.fn @"match_tuple_subject
# CHECK:         lit.call {{.*}}@"__getitem_param__
# CHECK:         lit.ref.load
# CHECK:         lit.call {{.*}}@"__eq__(
# CHECK:         lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:       [[AND:%.*]] = hlcf.elif %{{.*}} -> !kgen.scalar<bool> {
# CHECK:           lit.call {{.*}}@"__getitem_param__
# CHECK:           lit.ref.load
# CHECK:           lit.call {{.*}}@"__eq__(
# CHECK:           lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:           hlcf.yield {{.*}} : !kgen.scalar<bool>
# CHECK:       } else {
# CHECK:           kgen.param.constant: scalar<bool> = <false>
# CHECK:           hlcf.yield {{.*}} : !kgen.scalar<bool>
# CHECK:       }
# CHECK:       hlcf.elif [[AND]] {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 0
def match_tuple_subject(point: Tuple[Int, Int]):
    __match point:
    case (0, 0):
        case_callee[0]()
    case _:
        case_callee[1]()


# CHECK-LABEL: lit.fn @"match_with_guard
# CHECK-NEXT:    [[Z0:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 0}>
# CHECK-NEXT:    [[NE0:%.*]] = lit.call {{.*}}@"__ne__({{.*}}(%c, [[Z0]])
# CHECK-NEXT:    [[B0:%.*]] = lit.call {{.*}}@"__mlir_bool__(::Bool)"([[NE0]])
# CHECK-NEXT:    hlcf.elif [[B0]] {
# CHECK-NEXT:      lit.call {{.*}}@"case_callee{{.*}}<index> 0
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    } else {
# CHECK-NEXT:      [[L1:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 0}>
# CHECK-NEXT:      [[EQ1:%.*]] = lit.call {{.*}}@"__eq__({{.*}}(%x, [[L1]])
# CHECK-NEXT:      [[B1:%.*]] = lit.call {{.*}}@"__mlir_bool__(::Bool)"([[EQ1]])
# CHECK-NEXT:      [[G1:%.*]] = hlcf.elif [[B1]] -> !kgen.scalar<bool> {
# CHECK-NEXT:        [[Z1:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 0}>
# CHECK-NEXT:        [[NE1:%.*]] = lit.call {{.*}}@"__ne__({{.*}}(%c, [[Z1]])
# CHECK-NEXT:        [[GB1:%.*]] = lit.call {{.*}}@"__mlir_bool__(::Bool)"([[NE1]])
# CHECK-NEXT:        hlcf.yield [[GB1]] : !kgen.scalar<bool>
# CHECK-NEXT:      } else {
# CHECK-NEXT:        [[F1:%.*]] = kgen.param.constant: scalar<bool> = <false>
# CHECK-NEXT:        hlcf.yield [[F1]] : !kgen.scalar<bool>
# CHECK-NEXT:      }
# CHECK-NEXT:      hlcf.elif.yield [[G1]]
# CHECK-NEXT:    } then {
# CHECK-NEXT:      lit.call {{.*}}@"case_callee{{.*}}<index> 1
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    } else {
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    }
def match_with_guard(x: Int, c: Int):
    __match x:
    case _ if c != 0:
        case_callee[0]()
    case 0 if c != 0:
        case_callee[1]()


# CHECK-LABEL: lit.fn @"match_case_body
# CHECK:         kgen.param.constant: !Int = <{:scalar<index> 0}>
# CHECK:         lit.call {{.*}}@"__eq__(
# CHECK:         lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:       hlcf.elif %{{.*}} {
# CHECK:         %inside_case = lit.var.decl "inside_case"
# CHECK:         hlcf.yield
# CHECK:       } else {
# CHECK:         kgen.param.constant: scalar<bool> = <true>
# CHECK:         hlcf.elif.yield
# CHECK:       } then {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 0
# CHECK:         hlcf.yield
# CHECK:       } else {
# CHECK:         hlcf.yield
# CHECK:       }
def match_case_body(x: Int):
    __match x:
    case 0:
        var inside_case: Int
    case _:
        case_callee[0]()


# CHECK-LABEL: lit.fn @"match_float
# CHECK:         lit.call {{.*}}@"__eq__(
# CHECK:         lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:       hlcf.elif %{{.*}} {
def match_float(x: Float64):
    __match x:
    case 0.0:
        case_callee[0]()
    case _:
        case_callee[1]()


# CHECK-LABEL: lit.fn @"match_string
# CHECK:         lit.call {{.*}}@"__eq__(
# CHECK:         lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:       hlcf.elif %{{.*}} {
def match_string(x: String):
    __match x:
    case "a":
        case_callee[0]()
    case _:
        case_callee[1]()


# CHECK-LABEL: lit.fn @"match_bool
# CHECK:         lit.call {{.*}}@"__eq__(
# CHECK:         lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:       hlcf.elif %{{.*}} {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 0
# CHECK:         hlcf.yield
# CHECK:       } else {
# CHECK:         lit.call {{.*}}@"__eq__(
# CHECK:         lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:         hlcf.elif.yield
# CHECK:       } then {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 1
# CHECK:         hlcf.yield
# CHECK:       } else {
def match_bool(x: Bool):
    __match x:
    case True:
        case_callee[0]()
    case False:
        case_callee[1]()
    case _:
        case_callee[2]()


# Enum-like Color with inferred-member patterns (e.g. `case .red`).
struct Color(ImplicitlyCopyable):
    comptime red = Color()
    comptime green = Color()
    comptime blue = Color()

    def __init__(out self):
        pass

    def __eq__(self, other: Self) -> Bool:
        return True


# CHECK-LABEL: lit.fn @"match_color
# CHECK:         lit.call {{.*}}@"__eq__(
# CHECK:         lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:       hlcf.elif %{{.*}} {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 0
# CHECK:         hlcf.yield
# CHECK:       } else {
# CHECK:         lit.call {{.*}}@"__eq__(
# CHECK:         lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:         hlcf.elif.yield
# CHECK:       } then {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 1
# CHECK:         hlcf.yield
# CHECK:       } else {
# CHECK:         lit.call {{.*}}@"__eq__(
# CHECK:         lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:         hlcf.elif.yield
# CHECK:       } then {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 2
# CHECK:         hlcf.yield
# CHECK:       } else {
# CHECK:         kgen.param.constant: scalar<bool> = <true>
# CHECK:         hlcf.elif.yield
# CHECK:       } then {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 3
# CHECK:         hlcf.yield
# CHECK:       } else {
# CHECK:         hlcf.yield
# CHECK:       }
def match_color(c: Color):
    __match c:
    case Color.red:  # explicit enum case.
        case_callee[0]()
    case .green:     # inferred case.
        case_callee[1]()
    case .blue:     # inferred case.
        case_callee[2]()
    case _:
        case_callee[3]()


# CHECK-LABEL: lit.fn @"match_var_and_ref_binding
def match_var_and_ref_binding(value: String, mut mutString: String):
    # `var x` copies the borrowed subject into an owned binding that can be mutated.
    # CHECK:       [[X:%.*]] = lit.var.decl "x" var
    # CHECK:       lit.call {{.*}}@"__init__(copy:::String)"{{.*}}(%value, [[X]])
    # CHECK:       lit.call {{.*}}@"__iadd__{{.*}}([[X]],
    # CHECK:       lit.call {{.*}}@"byte_length(
    __match value:
    case var x:
        x += "x"
        _ = x.byte_length()

    # `ref y` stores a reference to the subject; no copy.
    # CHECK:       [[Y:%.*]] = lit.var.decl "y" ref
    # CHECK:       lit.ref.store %value, [[Y]]
    # CHECK:       lit.call {{.*}}@"byte_length(
    __match value:
    case ref y:
        _ = y.byte_length()

    # `ref z` maintains the mutability of the subject.
    # CHECK:       [[Z:%.*]] = lit.var.decl "z" ref
    # CHECK:       lit.ref.store %mutString, [[Z]]
    # CHECK:       lit.call {{.*}}@"__iadd__{{.*}}(
    __match mutString:
    case ref z:
        z += "x"


# Tuple patterns with nested var bindings (from the pattern-matching proposal).
def match_inspect_point(point: Tuple[Int, Int]):
    __match point:
    case (0, 0):
        case_callee[0]()
    case (var x, 0):
        case_callee[1]()
    case (0, var y):
        case_callee[2]()
    case (var x, var y):
        case_callee[3]()


# CHECK-LABEL: lit.fn @"match_as_pattern
def match_as_pattern(value: String):
    # `as` binds a ref to the whole subject, never a copy.
    # CHECK:       [[S:%.*]] = lit.var.decl "s" ref
    # CHECK:       lit.ref.store %value, [[S]]
    # CHECK-NOT:   lit.call {{.*}}@"__init__(copy:::String)"
    # CHECK:       lit.call {{.*}}@"byte_length(
    __match value:
    case _ as s:
        _ = s.byte_length()

    # Combined with a value pattern: still a ref, plus an equality test.
    # CHECK:       [[T:%.*]] = lit.var.decl "t" ref
    # CHECK:       lit.ref.store %value, [[T]]
    # CHECK:       lit.call {{.*}}@"__eq__(
    __match value:
    case "hello" as t:
        _ = t.byte_length()

    # Register-passable subjects cannot be `ref`; `as` uses `bind` instead.
    # CHECK:       [[X:%.*]] = lit.var.decl "x" bound
    # CHECK:       lit.ref.store {{.*}}, [[X]]
    # CHECK:       lit.call {{.*}}@"__add__(
    __match 42:
    case _ as x:
        _ = x + 1

    # The first case's bindings are emitted in place; later cases are hoisted
    # above the elif so they dominate their condition and body regions.
    # CHECK:       [[ORIGIN:%.*]] = lit.var.decl "origin" ref
    # CHECK:       lit.ref.store %point, [[ORIGIN]]
    # CHECK:       [[P:%.*]] = lit.var.decl "p" ref
    # CHECK:       lit.var.decl "x" var
    # CHECK:       hlcf.elif %{{.*}} {
    # CHECK:       lit.ref.store %point, [[P]]
    var point: Tuple[Int, Int] = (0, 0)
    __match point:
    case (0, 0) as origin:
        _ = origin
        case_callee[0]()
    case (var x, 0) as p:
        _ = x
        _ = p
        case_callee[1]()


# CHECK-LABEL: lit.fn @"match_or_pattern_int
def match_or_pattern_int(x: Int):
    # CHECK-NEXT:    [[L0:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 0}>
    # CHECK-NEXT:    [[EQ0:%.*]] = lit.call {{.*}}@"__eq__({{.*}}(%x, [[L0]])
    # CHECK-NEXT:    [[B0:%.*]] = lit.call {{.*}}@"__mlir_bool__(::Bool)"([[EQ0]])
    # CHECK-NEXT:    [[OR01:%.*]] = hlcf.elif [[B0]] -> !kgen.scalar<bool> {
    # CHECK-NEXT:      [[T01:%.*]] = kgen.param.constant: scalar<bool> = <true>
    # CHECK-NEXT:      hlcf.yield [[T01]] : !kgen.scalar<bool>
    # CHECK-NEXT:    } else {
    # CHECK-NEXT:      [[L1:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 1}>
    # CHECK-NEXT:      [[EQ1:%.*]] = lit.call {{.*}}@"__eq__({{.*}}(%x, [[L1]])
    # CHECK-NEXT:      [[B1:%.*]] = lit.call {{.*}}@"__mlir_bool__(::Bool)"([[EQ1]])
    # CHECK-NEXT:      hlcf.yield [[B1]] : !kgen.scalar<bool>
    # CHECK-NEXT:    }
    # CHECK-NEXT:    hlcf.elif [[OR01]] {
    # CHECK-NEXT:      lit.call {{.*}}@"case_callee{{.*}}<index> 0
    # CHECK-NEXT:      hlcf.yield
    # CHECK-NEXT:    } else {
    # CHECK-NEXT:      [[L2:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 0}>
    # CHECK-NEXT:      [[EQ2:%.*]] = lit.call {{.*}}@"__eq__({{.*}}(%x, [[L2]])
    # CHECK-NEXT:      [[B2:%.*]] = lit.call {{.*}}@"__mlir_bool__(::Bool)"([[EQ2]])
    # CHECK-NEXT:      [[OR01:%.*]] = hlcf.elif [[B2]] -> !kgen.scalar<bool> {
    # CHECK-NEXT:        [[T01:%.*]] = kgen.param.constant: scalar<bool> = <true>
    # CHECK-NEXT:        hlcf.yield [[T01]] : !kgen.scalar<bool>
    # CHECK-NEXT:      } else {
    # CHECK-NEXT:        [[L3:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 1}>
    # CHECK-NEXT:        [[EQ3:%.*]] = lit.call {{.*}}@"__eq__({{.*}}(%x, [[L3]])
    # CHECK-NEXT:        [[B3:%.*]] = lit.call {{.*}}@"__mlir_bool__(::Bool)"([[EQ3]])
    # CHECK-NEXT:        hlcf.yield [[B3]] : !kgen.scalar<bool>
    # CHECK-NEXT:      }
    # CHECK-NEXT:      [[OR012:%.*]] = hlcf.elif [[OR01]] -> !kgen.scalar<bool> {
    # CHECK-NEXT:        [[T012:%.*]] = kgen.param.constant: scalar<bool> = <true>
    # CHECK-NEXT:        hlcf.yield [[T012]] : !kgen.scalar<bool>
    # CHECK-NEXT:      } else {
    # CHECK-NEXT:        [[L4:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 2}>
    # CHECK-NEXT:        [[EQ4:%.*]] = lit.call {{.*}}@"__eq__({{.*}}(%x, [[L4]])
    # CHECK-NEXT:        [[B4:%.*]] = lit.call {{.*}}@"__mlir_bool__(::Bool)"([[EQ4]])
    # CHECK-NEXT:        hlcf.yield [[B4]] : !kgen.scalar<bool>
    # CHECK-NEXT:      }
    # CHECK-NEXT:      hlcf.elif.yield [[OR012]]
    # CHECK-NEXT:    } then {
    # CHECK-NEXT:      lit.call {{.*}}@"case_callee{{.*}}<index> 1
    # CHECK-NEXT:      hlcf.yield
    # CHECK-NEXT:    } else {
    # CHECK-NEXT:      hlcf.yield
    # CHECK-NEXT:    }
    __match x:
    case 0 | 1:
        case_callee[0]()
    case 0 | 1 | 2:
        case_callee[1]()

# CHECK-LABEL: lit.fn @"match_or_pattern_tup
def match_or_pattern_tup(point: Tuple[Int, Int]):
    # CHECK-NEXT:    lit.call {{.*}}@"__getitem_param__
    # CHECK:         lit.call {{.*}}@"__eq__(
    # CHECK:         [[AND:%.*]] = hlcf.elif %{{.*}} -> !kgen.scalar<bool> {
    # CHECK:           lit.call {{.*}}@"__getitem_param__
    # CHECK:           lit.call {{.*}}@"__eq__(
    # CHECK:           hlcf.yield {{.*}} : !kgen.scalar<bool>
    # CHECK:         } else {
    # CHECK:           kgen.param.constant: scalar<bool> = <false>
    # CHECK:           hlcf.yield {{.*}} : !kgen.scalar<bool>
    # CHECK:         }
    # CHECK:         [[OR:%.*]] = hlcf.elif [[AND]] -> !kgen.scalar<bool> {
    # CHECK:           kgen.param.constant: scalar<bool> = <true>
    # CHECK:           hlcf.yield {{.*}} : !kgen.scalar<bool>
    # CHECK:         } else {
    # CHECK:           lit.call {{.*}}@"__getitem_param__
    # CHECK:           lit.call {{.*}}@"__eq__(
    # CHECK:           hlcf.yield {{.*}} : !kgen.scalar<bool>
    # CHECK:         }
    # CHECK:       hlcf.elif [[OR]] {
    # CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 2
    __match point:
    case (0, 0) | (1, 1):
        case_callee[2]()

# CHECK-LABEL: lit.fn @"match_or_pattern_bind
def match_or_pattern_bind(var point: Tuple[Int, Int]):
    # Both alternatives bind `x`; one VarDecl is shared.
    # CHECK:       [[X:%.*]] = lit.var.decl "x" var
    # CHECK:       lit.ref.store {{.*}}, [[X]]
    # CHECK:       lit.call {{.*}}@"case_callee{{.*}}<index> 3
    __match point:
    case (0, var x) | (var x, 1):
        _ = x
        case_callee[3]()

    # Same with `ref` bindings.
    # CHECK:       [[RX:%.*]] = lit.var.decl "x" ref
    # CHECK:       lit.call {{.*}}@"case_callee{{.*}}<index> 4
    __match point:
    case (2, ref x) | (ref x, 3):
        _ = x
        case_callee[4]()

@fieldwise_init
struct Vec3:
    var x: Int
    var y: Int
    var z: Int


# CHECK-LABEL: lit.fn @"match_vec3
def match_vec3(v: Vec3):
    # Keyword field patterns: project each named field and match it.
    # CHECK:       lit.ref.struct.ger {{.*}}[x]
    # CHECK:       lit.call {{.*}}@"__eq__(
    # CHECK:       lit.ref.struct.ger {{.*}}[y]
    # CHECK:       lit.call {{.*}}@"__eq__(
    # CHECK:       lit.ref.struct.ger {{.*}}[z]
    # CHECK:       lit.call {{.*}}@"__eq__(
    __match v:
    case Vec3(x=0, y=0, z=0):
        case_callee[0]()

    # Bind one field, test another, ignore the third.
    # CHECK:       lit.ref.struct.ger {{.*}}[x]
    # CHECK:       [[X:%.*]] = lit.var.decl "x" var
    # CHECK:       lit.ref.store {{.*}}, [[X]]
    # CHECK:       lit.ref.struct.ger {{.*}}[y]
    # CHECK:       lit.call {{.*}}@"__eq__(
    __match v:
    case Vec3(x=var x, y=0, z=_):
        _ = x
        case_callee[1]()

    # Positional subpatterns bind stored fields in declaration order.
    # CHECK:       lit.var.decl "x" var
    # CHECK:       lit.var.decl "y" var
    # CHECK:       lit.var.decl "z" var
    __match v:
    case var Vec3(x, y, z):
        _ = x + y + z
        case_callee[2]()


# CHECK-LABEL: lit.fn @"match_optional
def match_optional(opt: Optional[Int], mut mut_opt: Optional[Int]):
    # Immutable Optional: discriminant check, then payload projection + bind.
    # CHECK:       lit.call {{.*}}@"_get_enum_discriminant{{.*}}[imm *"opt`
    # CHECK:       lit.call {{.*}}@"__eq__(
    # CHECK:       [[ELT:%.*]] = lit.var.decl "elt" ref
    # CHECK:       lit.call {{.*}}@"_unsafe_get_enum_payload{{.*}}(%opt)
    # CHECK:       lit.ref.store {{.*}}, [[ELT]]
    __match opt:
    case Optional.Some(ref elt):
        _ = elt
        case_callee[0]()
    # CHECK:       lit.call {{.*}}@"_get_enum_discriminant{{.*}}[imm *"opt`
    # CHECK:       lit.call {{.*}}@"__eq__(
    # CHECK:       lit.call {{.*}}@"case_callee{{.*}}<index> 1
    case Optional.None:
        case_callee[1]()

    # Mutable Optional: same match path; subject origin is mut (muttoimm on
    # read-only EnumLike accessors until those use an interior origin).
    # CHECK:       lit.ref.immut %mut_opt
    # CHECK:       lit.call {{.*}}@"_get_enum_discriminant{{.*}}[muttoimm *"mut_opt`
    # CHECK:       lit.call {{.*}}@"__eq__(
    # CHECK:       [[MELT:%.*]] = lit.var.decl "elt" ref
    # CHECK:       lit.call {{.*}}@"_unsafe_get_enum_payload{{.*}}muttoimm *"mut_opt`
    # CHECK:       lit.ref.store {{.*}}, [[MELT]]
    __match mut_opt:
    case Optional.Some(ref elt):
        _ = elt
        case_callee[2]()
    # CHECK:       lit.call {{.*}}@"_get_enum_discriminant{{.*}}[muttoimm *"mut_opt`
    # CHECK:       lit.call {{.*}}@"case_callee{{.*}}<index> 3
    case Optional.None:
        case_callee[3]()

    # Matching with inferred base also work.
    __match opt:
    case .Some(ref elt):
        case_callee[0]()
    case ((.None)):  # Extra parens are fine of course.
        case_callee[1]()
    case .Some:  # just check the tag, don't bind the value.
        case_callee[2]()
