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

comptime index42 = __mlir_attr.`42 : index`

def test_mlir():
  var x: __mlir_type.index
  x += x # expected-error {{'__mlir_type.index' does not implement the '__iadd__' method}}

  # expected-error @+1 {{'index.add' op expected 2 operands, but found 0}}
  __mlir_op.`index.add`()

  # expected-error @+1 {{'__mlir_type.index' has no attributes}}
  __mlir_op.`index.add`(x, x).x

  # expected-error @+1 {{operation already has attributes}}
  __mlir_op.`op`[value1=index42][value2=index42]

  __mlir_op.`op`[
    value=index42,  # expected-note {{previously specified here}}
    value=index42,  # expected-error {{keyword parameter 'value' was already used; remove the duplicate}}
  ]

def test_mlir2():
  # expected-error @below {{invalid MLIR type: kgen.dtype}}
  # expected-note @below {{MLIR error: expected non-function type}}
  var y : __mlir_type.`kgen.dtype`  # should be !kgen.dtype

  var a : __mlir_type
  var x: __mlir_type.index

  # expected-error @+1 {{unable to infer result type from MLIR operation 'index.castu'}}
  __mlir_op.`index.castu`(x, a)
  # expected-error @+1 {{unable to infer result type from MLIR operation 'index.castu'}}
  __mlir_op.`index.castu`(x)
  # expected-error @+1 {{'index.castu' op result #0 must be integer or index, but got 'f32'}}
  __mlir_op.`index.castu`[_type=__mlir_type.f32](x)

  # expected-error @below {{failed properties conversion while building index.constant with `{value = 4.200000e+01 : f32}`: Invalid attribute `value` in property conversion: 4.200000e+01 : f32}}
  # expected-error @below {{unable to infer result type from MLIR operation 'index.constant'}}
  var c42e = __mlir_op.`index.constant`[value=__mlir_attr.`42.0 : f32`]()
  var c42 = __mlir_op.`index.constant`[value=index42]() # Good

  # expected-error @below {{invalid MLIR attribute:}}
  # expected-note @below {{attempting to parse: '#index.cmp_predicate<xeq>'}}
  __mlir_attr.`#index.cmp_predicate<xeq>`

  # expected-warning @below {{'__mlir_type.`!kgen.deferred`' value is unused; assign to '_' to discard the result}}
  __mlir_attr.`#index.cmp_predicate<eq>`

  # expected-error @below {{invalid MLIR attribute: expected attribute value}}
  # expected-note @below {{attempting to parse: '_'}}
  __mlir_attr.

  # expected-error @below {{expected name in attribute reference}}
  _ = __mlir_op.`test.op`[__mlir_attr.]

  # expected-error @+1 {{cannot construct type '__mlir_type.index'}}
  _ = __mlir_type.index(index42)

  # expected-error @below {{invalid MLIR type: !kgen.pointer<!b>}}
  # expected-note @below {{undefined symbol alias id 'b'}}
  _ = __mlir_type[`!kgen.pointer<!b>`]


def colon_instead_of_equal():
  # expected-error @below {{attribute spec requires a keyword parameter; did you mean 'value=...'?}}
  _ = __mlir_op.`lit.crazy`[value:index42]()

struct Int(TrivialRegisterPassable):
  var value : __mlir_type.index

# Issue #7307: Error message can be improved when a user accidentally uses = instead of :
def equal_instead_of_colon():
  var someInt : Int
  # expected-error @+1 {{unable to infer result type from MLIR operation 'pop.array.gep'}}
  var ptr = __mlir_op.`pop.array.gep`((((someInt))), index42)

def crash_on_invalid():
  # expected-error @+1 {{use of unregistered MLIR operation 'invalid_op'}}
  _ = __mlir_op.`invalid_op`[_type=__mlir_type.i16]()


# expected-error @below {{invalid MLIR type}}
# expected-note @below {{argument #0 with convention 'mut' in func type should be a `!kgen.pointer`}}
def bad_signature_type[func: __mlir_type[`!kgen.func<(index mut) -> !kgen.none>`]]():
    pass


def mlir_magic_keyword_param():
    # expected-error @below {{only positional operands allowed in mlir magic}}
    comptime a = __mlir_type[a=`!kgen.scalar<bool>`]


def mlir_properties(arg0: __mlir_type.i64, arg1: __mlir_type.i64):
    _ = __mlir_op.`llvm.add`[
        _type = __mlir_type.i64,
        _properties = __mlir_attr.`#llvm.overflow<nsw>`,
    ](arg0, arg1)
    # expected-error @above {{cannot set property}}
    # expected-error @above {{expected DictionaryAttr to set properties}}


def mlir_illegal_op():
    # Intentional typo for `is_zero_poison`:
    # expected-error @below {{attribute 'is_zero_poson' is not an inherent attribute of 'llvm.intr.ctlz'}}
    __mlir_op.`llvm.intr.ctlz`[_type=Int, is_zero_poson=__mlir_attr.`0: i1`](1)

    # expected-error @below {{MLIR verification error: 'llvm.intr.ctlz' op requires attribute 'is_zero_poison'}}
    __mlir_op.`llvm.intr.ctlz`[_type=Int](1)
