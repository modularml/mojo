//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef MTYPES_TD
#define MTYPES_TD

include "Support/MDialect/MDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

class MType<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MDialect, typeName, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

def M_ArrayType : MType<"Array", "array",
    [DeclareTypeInterfaceMethods<ShapedTypeInterface>]> {
  let summary = "1D shaped type";
  let description = [{
    The `!M.array` type represents one dimensional data of known length. This
    type implements `ShapedType` and can be used with `ElementsAttr`.

    Example:

    ```mlir
    // An array of integers.
    !M.array<32xi32>

    // An array of floats.
    !M.array<256xf64>
    ```
  }];

  let parameters = (ins "int64_t":$size, "Type":$elementType);
  let assemblyFormat = "`<` custom<SizeX>($size) `` $elementType `>`";
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "int64_t":$size, "Type":$elementType)>
  ];
}

//===----------------------------------------------------------------------===//
// AlignedBytesType
//===----------------------------------------------------------------------===//

def M_AlignedBytesType : MType<"AlignedBytes", "aligned_bytes",
    [DeclareTypeInterfaceMethods<ShapedTypeInterface>]> {
  let summary = "Type of uint8_t arrays of known size and alignment.";
  let description = [{
    This type has no values and no runtime representation. It is intended only
    to be used as a type annotation on `dense_resource` attribute operands
    so as to convey a desired alignment. This is needed in two situations:
     - As a way to 'forward declare' the alignment for an attribute who's
       blob has not yet been parsed.
     - As a way to override the required alignment for an attribute without
       reallocating the underlying data.

    Example:
    ```mlir
    // An array of 4 uint8_ts with 16 byte alignment
    !M.aligned_bytes<4, align 16>
    ```
  }];

  let parameters = (ins "uint64_t":$size, "uint64_t":$align);
  let assemblyFormat = "`<` $size `,` `align` $align `>`";
}

#endif // MTYPES_TD
