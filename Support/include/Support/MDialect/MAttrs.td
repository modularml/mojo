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

#ifndef MATTRS_TD
#define MATTRS_TD

include "Support/MDialect/MDialect.td"
include "Support/MDialect/MAttrInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

class MAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<MDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// StringArrayAttr
//===----------------------------------------------------------------------===//

def StringArrayAttr
    : ArrayOfAttr<MDialect, "StringArray", "strings", "StringAttr">;

//===----------------------------------------------------------------------===//
// SymbolRefArrayAttr
//===----------------------------------------------------------------------===//

def M_SymbolRefArrayAttr
    : ArrayOfAttr<MDialect, "SymbolRefArray", "symbols", "SymbolRefAttr">;

//===----------------------------------------------------------------------===//
// TypeArrayAttr
//===----------------------------------------------------------------------===//

// This is an array of type objects, stored in a dense representation that can
/// be turned into an ArrayRef.
/// NOTE: We have to define this as M_TypeArrayAttr because ODS squats on the
/// `TypeArrayAttr` name with something different.
def M_TypeArrayAttr : ArrayOfAttr<MDialect, "TypeArray", "type.array", "Type">;

//===----------------------------------------------------------------------===//
// PrimitivesArrayAttr
//===----------------------------------------------------------------------===//

def M_PrimitiveArrayData : ArrayRefParameter<"uint8_t",
    "aligned storage for primitive elements"> {
  let allocator = [{
    $_dst = copyIntoAlignedBuffer($_allocator, $_self, elementType);
  }];
  let cppStorageType = "std::vector<uint8_t>";
}

def M_PrimitiveArrayAttr : MAttr<"PrimitiveArray", "primitives_array"> {
  let summary = "an array of primitive elements";
  let description = [{
    The `#M.primitives_array` attribute represents an array of primitive
    (boolean, integer, index, or floating point) data of equal size. The data is
    stored as a byte array with element types whose sizes are not multiples of
    bytes padded to the nearest byte. The underlying array is aligned to that
    byte size.

    Example:

    ```mlir
    // An array of integers.
    #M.primitives_array<si24: -2, 0, 2>

    // An array of floats.
    #M.primitives_array<bf16: 0.2, 1.2, 3.>

    // Boolean arrays use `i1` elements.
    #M.primitives_array<i1: true, false, true>
    ```
  }];

  let parameters = (ins M_PrimitiveArrayData:$data, "Type":$elementType);
  let assemblyFormat = [{
    `<` $elementType `` custom<PrimitiveArray>($data, ref($elementType)) `>`
  }];
  let genVerifyDecl = 1;

  let builders = [
    AttrBuilderWithInferredContext<(ins "ArrayRef<uint8_t>":$data,
                                        "Type":$elementType)>
  ];

  let extraClassDeclaration = [{
    /// Return the number of elements.
    int64_t size() const;
  }];
}

//===----------------------------------------------------------------------===//
// ArrayElementsAttr
//===----------------------------------------------------------------------===//

def M_ArrayElementsAttr : MAttr<"ArrayElements", "dense_array",
    [ElementsAttrInterface, TypedAttrInterface,
     DeclareAttrInterfaceMethods<HasAlignedBytesInterface, ["getAlignedBytesType"]>]> {
  let summary = "an elements attribute backed by an array";
  let description = [{
    The `#M.dense_array` attribute is an elements attribute backed by a
    primitive array. The attribute supports the full API expected by
    `ElementsAttr` and users thereof, including arbitrary shaped types and
    failable value iteration.

    Importantly, this attribute can only contain scalar integer and floating
    point types, does not bit-pack elements, and does not have special handling
    for splat elements.

    This attribute implements the HasAlignedBytesInterface by returning a
    !M.aligned_bytes type capturing the byte size of the underlying element data
    array and an alignment of the next power-of-two of the byte size of the
    element type.

    Example:

    ```mlir
    // An array of integers.
    #M.dense_array<-3, 0, 1, 42> : !M.array<4xi32>

    // A float tensor.
    #M.dense_array<3.4, 2.3, 5.2, 1.9> : tensor<2x2xf32>

    // A 0D vector.
    #M.dense_array<1> : vector<ui64>
    ```
  }];

  let parameters = (ins "PrimitiveArrayAttr":$data,
                        AttributeSelfTypeParameter<"", "ShapedType">:$type);
  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    AttrBuilderWithInferredContext<(ins "ArrayRef<uint8_t>":$data,
                                        "ShapedType":$type)>
  ];

  let extraClassDeclaration = [{
    /// Get the raw underlying data.
    ArrayRef<uint8_t> getRawData() const;

    /// `ElementsAttr` implementation. Support iteration with `Attribute` to be
    /// compatible with upstream dialects.
    using NonContiguousIterableTypesT = std::tuple<Attribute>;
    FailureOr<detail::AttrIterator>
    try_value_begin_impl(OverloadToken<Attribute>) const;
  }];
}

//===----------------------------------------------------------------------===//
// AlignedBytesAttr
//===----------------------------------------------------------------------===//

// Make sure the attribute's data argument is interned with the requested
// alignment.
def M_AlignedBytesData : ArrayRefParameter<"uint8_t",
    "aligned storage for AlignedBytesAttr"> {
  let allocator = [{
    $_dst = copyIntoBytes($_allocator, $_self, align);
  }];
}

def M_AlignedBytesAttr : MAttr<"AlignedBytes", "aligned_bytes", [
    DeclareAttrInterfaceMethods<ElementsAttrInterface>,
    DeclareAttrInterfaceMethods<HasAlignedBytesInterface, ["getAlignedBytesType"]>]> {
  let summary = "A literal byte array with alignment.";
  let description = [{
    This attribute is the 'inline' equivalent of `dense_resource`, and describes
    a byte array with byte alignment. Nothing is implied about the internal
    structure of the bytes (rank, dimensions, underlying element type, padding,
    or element endianness).

    This attribute implements the ElementsAttrInterface as a rank-1 ShapedType
    over uint8_t elements.

    This attribute implements the HasAlignedBytesInterface by returning the
    obvious !M.aligned_bytes type.

    Generally this attribute should only be used for 'small' literals. Larger
    literals should be placed into the resources table and be referenced by the
    `dense_resources` attribute. Note that dense resources encode their
    alignment in their first four bytes using the little-endian convention,
    where as this attribute describes the alignment explicitly.

    Examples:
    ```mlir
    #M.aligned_bytes<"0x01020304", align 64> // 4 bytes, aligned on 64 bytes
    ```
  }];

  let parameters = (ins M_AlignedBytesData:$data, "uint64_t":$align);
  let assemblyFormat = [{
    `<` custom<AlignedBytesData>($data) `,` `align` $align `>`
  }];
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// ElementsAttr interface implementation.
    using ContiguousIterableTypesT = std::tuple<uint8_t>;
    FailureOr<const uint8_t*> try_value_begin_impl(OverloadToken<uint8_t>) const;
    ShapedType getType() const;
  }];
}

//===----------------------------------------------------------------------===//
// IntArrayElementsAttr
//===----------------------------------------------------------------------===//

def IntArrayElementsAttr : Attr<CPred<"isa<IntArrayElementsAttr>($_self)">,
    "Dense integer array"> {
  let storageType = "IntArrayElementsAttr";
  let returnType = "IntArrayElementsAttr";
}

def SI32ArrayElementsAttr : ConfinedAttr<IntArrayElementsAttr, [AttrConstraint<CPred<
    "cast<IntArrayElementsAttr>($_self).getElementType().isSignedInteger(32)"
>>]> {
  let constBuilderCall = "$_builder.getAttr<IntArrayElementsAttr>($0, IntegerType::Signed)";
  let returnType = "ArrayRef<int32_t>";
  let convertFromStorage = "$_self.asArrayRef<int32_t>()";
}

def SI64ArrayElementsAttr : ConfinedAttr<IntArrayElementsAttr, [AttrConstraint<CPred<
    "cast<IntArrayElementsAttr>($_self).getElementType().isSignedInteger(64)"
>>]> {
  let constBuilderCall = "$_builder.getAttr<IntArrayElementsAttr>($0, IntegerType::Signed)";
  let returnType = "ArrayRef<int64_t>";
  let convertFromStorage = "$_self.asArrayRef<int64_t>()";
}

def UI8ArrayElementsAttr : ConfinedAttr<IntArrayElementsAttr, [AttrConstraint<CPred<
    "cast<IntArrayElementsAttr>($_self).getElementType().isUnsignedInteger(8)"
>>]> {
  let constBuilderCall = "$_builder.getAttr<IntArrayElementsAttr>($0, IntegerType::Unsigned)";
  let returnType = "ArrayRef<uint8_t>";
  let convertFromStorage = "$_self.asArrayRef<uint8_t>()";
}

def UI64ArrayElementsAttr : ConfinedAttr<IntArrayElementsAttr, [AttrConstraint<CPred<
    "cast<IntArrayElementsAttr>($_self).getElementType().isUnsignedInteger(64)"
>>]> {
  let constBuilderCall = "$_builder.getAttr<IntArrayElementsAttr>($0, IntegerType::Unsigned)";
  let returnType = "ArrayRef<uint64_t>";
  let convertFromStorage = "$_self.asArrayRef<uint64_t>()";
}

//===----------------------------------------------------------------------===//
// MultiLineString
//===----------------------------------------------------------------------===//

def M_MultiLineStringAttr
    : ArrayOfAttr<MDialect, "MultiLineString", "multiline", "StringAttr"> {
  let summary = "An array of strings which is encoded as a multi-line "
                "null-terminated string in MEF.";
  let description = [{
    An array of strings.

    Encoded into MEF as a multi-line null-terminated string using `\n` as
    the line terminator.

    Example: The attribute
    ```mlir
      #m<multiline["line1",
                   "line2"]>
    ```
    will appear in MEF as the the string "line1\nline2\n\0".
  }];
}

//===----------------------------------------------------------------------===//
// TargetInfoAttr
//===----------------------------------------------------------------------===//

def OptionalStringRefParameter : StringRefParameter<""> {
  let defaultValue = "\"\"";
}

def OptionalDataLayoutParameter : AttrParameter<"DataLayout", "data layout", "const DataLayout &"> {
  let defaultValue = "DataLayout()";
}

def M_TargetInfoAttr : MAttr<"TargetInfo", "target"> {
  let summary = "Target configuration attribute.";
  let description = [{
    The `#M.target` attribute represents a compilation target configuration. It
    contains the target triple, microarchitecture, stdlib plugin, optional features,
    and derived data such as data layout and SIMD width.

    The user can also specify the default index bit width being used. This is
    important for GPUs where the hardware can support different index bit widths
    for cross compilation.

    The microarchitecture and features may encode version numbers, such as
    'sm_80' for CUDA compute capability 8.0. When deciding if a runtime host
    supports a compile time microarchitecutre or feature it may be necessary
    to compare these versions numerically rather than textually.

    Features are encoded in the form of "+feature1,+feature2,-feature3",
    where "+" enables and "-" disables a feature. The list is fully
    expanded: it includes both the features explicitly requested by the
    user and all features enabled by default for the target CPU (e.g.
    avx512f on znver4), with explicit user overrides applied on top.
    A feature present in the list without a "-" entry is enabled; a
    feature absent from the list was not enabled by the CPU model and
    was not explicitly requested.

    Example:
    ```mlir
    #M.target<triple="x86_64-unknown-linux-gnu", arch="znver3",
              stdlib_plugin="default", features="+avx,+avx2",
              data_layout="p:64:64-i64:64:64", relocation_model="static",
              simd_bit_width=256, index_bit_width=64>
    #M.target<triple="nvptx64-nvidia-cuda", arch="sm_80", stdlib_plugin="cuda">
    ```

    Can be (partially) represented at runtime by M::TargetInfo.

    The `accelerator_arch` contains the vendor name of the accelerator in
    lowercase (e.g. nvidia or amd) along with the compute capability (e.g. 80 or
    90). For example, a valid `accelerator_arch` is `nvidia:80` or `amdgpu:gfx942`.

    The `abi` names the target ABI (e.g. RISC-V's `lp64d`, MIPS's `n32`),
    mirroring clang's `-target-abi`. When set, it is recorded as a
    `target-abi` LLVM module flag, exactly as clang does.
  }];

  let parameters = (ins
    "llvm::Triple":$triple,
    StringRefParameter<>:$arch,
    StringRefParameter<"", "\"default\"">:$stdlib_plugin,
    OptionalStringRefParameter:$features,
    OptionalDataLayoutParameter:$data_layout,
    DefaultValuedParameter<"llvm::Reloc::Model",
                           "llvm::Reloc::Static">:$relocation_model,
    DefaultValuedParameter<"int64_t", "0">:$simd_bit_width,
    DefaultValuedParameter<"std::optional<int64_t>",
                           "std::nullopt">:$index_bit_width,
    OptionalStringRefParameter:$tune_cpu,
    OptionalStringRefParameter:$accelerator_arch,
    OptionalStringRefParameter:$abi
  );

  let assemblyFormat = "`<` struct(params) `>`";

  let extraClassDeclaration = [{
    /// Get the target triple as a string.
    std::string getTripleStr() const {
      return getTriple().normalize();
    }

    /// Returns whether the triple is a GPU target.
    bool isGPU() const {
      return getTriple().isNVPTX() || getTriple().isAMDGPU() || isMetal();
    }

    /// Returns whether this is a Metal target (macOS with actual GPU, not embedded devices).
    bool isMetal() const {
      // Use the same logic as isMetalTriple() for consistency
      // Accept both air64 and arm64 triples for Metal GPU targets
      return getTripleStr().starts_with("air64");
    }

    /// Check if the target supports a specific hardware feature.
    /// See M::hasFeature for the matching contract.
    bool hasFeature(StringRef feature) const {
      return M::hasFeature(getFeatures(), feature);
    }

    /// Returns the operating system for the triple.
    StringRef getOS() const {
      return llvm::Triple::getOSTypeName(getTriple().getOS());
    }

    /// Returns the resolved index_bit_width.
    int64_t resolveIndexBitWidth() {
      return getIndexBitWidth().value_or(getDataLayout().getPointerBitWidth());
    }

    /// Returns ptr byteSize for the interpreter which is always 8
    int32_t getDefaultPointerSize() {
      return 8;
    }
  }];
}

//===----------------------------------------------------------------------===//
// DeviceRefAttr
//===----------------------------------------------------------------------===//

def M_DeviceRefAttr : MAttr<"DeviceRef", "device_ref"> {
  let summary = "Refers to a device spec a value or operation occupies";
  let description = [{
    The `#M.device_ref` attribute refers to a unique `#M.device_spec` within the
    overall model `#M.device_spec_collection`. It contain a label and id.

    Example:
    ```mlir
      #M.device_ref<"gpu", 0>
    ```
  }];

  let parameters = (ins
    StringRefParameter<>:$label,
    "DeviceId":$id
  );

  let constBuilderCall = "M::DeviceRefAttr::get($_builder.getContext(), $0)";

  let assemblyFormat = [{
    `<` $label `,` $id `>`
  }];

  let extraClassDeclaration = [{
    /// Returns device ref in compact string form.
    std::string toString() const;

    /// Returns the default M.device_ref<"cpu", 0> device ref.
    static DeviceRefAttr getDefault(mlir::MLIRContext* ctx);
  }];
}

//===----------------------------------------------------------------------===//
// DeviceRefArrayAttr
//===----------------------------------------------------------------------===//

def M_DeviceRefArrayAttr : TypedArrayAttrBase<M_DeviceRefAttr, "A list of devicerefs">;

//===----------------------------------------------------------------------===//
// DeviceInfoAttr
//===----------------------------------------------------------------------===//

def M_DeviceInfoAttr : MAttr<"DeviceInfo", "device_info"> {
  let summary = "Runtime-probed information about a physical device";
  let description = [{
    The `#M.device_info` attribute captures runtime-probed properties of a
    physical device: its label (e.g. `"gpu"`), the underlying API
    (e.g. `"cuda"`, `"hip"`), the architecture name
    (e.g. `"sm_100a"`, `"gfx942"`), and the model name
    (e.g. `"NVIDIA B200"`). This is used to drive per-kernel device
    registration and specialization.

    Example:
    ```mlir
      #M.device_info<"gpu", "cuda", "sm_100a", "NVIDIA B200">
    ```
  }];

  let parameters = (ins
    StringRefParameter<>:$label,
    StringRefParameter<>:$api,
    StringRefParameter<>:$arch,
    StringRefParameter<>:$model,
    DefaultValuedParameter<"bool", "false">:$tile_based_fusion
  );

  let assemblyFormat = [{
    `<` $label `,` $api `,` $arch `,` $model
    (`,` `tile_based_fusion` `=` $tile_based_fusion^)? `>`
  }];
}

//===----------------------------------------------------------------------===//
// DeviceSpecAttr
//===----------------------------------------------------------------------===//

def M_DeviceSpecAttr : MAttr<"DeviceSpec", "device_spec"> {
  let summary = "Compile-time information about a computational device.";
  let description = [{
    The `#M.device_spec` attribute describes everything we need to know about a
    computational device at compile time which is expected to be present at
    runtime. This includes the device's `#M.target` configuration (to guide
    code generation), the device's id (to disambiguate the device from other
    devices with similar target configuration, default 0), and a string label
    (so model types and ops may refer to a specific device when needed, eg for
    device placement).

    Device ids need not refer to physical devices (eg a specific CUDA device
    id), they are just placeholders. It is the responsibility of the various
    setup ops in the model's 'init' block to establish the mapping from device
    ids used in the model to actual physical devices where required. This may
    require matching properties of the `#M.target` to capabilities probed at
    runtime.

    ```mlir
    #M.device_spec<ref = <"gpu", 1>,
                   target = <triple="nvptx64-nvidia-cuda", arch="sm_80">>
    ```

    Can be represented at runtime by M::DeviceSpec.
  }];

  let parameters = (ins
    M_DeviceRefAttr:$ref,
    M_TargetInfoAttr:$target
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// DeviceSpecCollectionAttr
//===----------------------------------------------------------------------===//

def M_DeviceSpecCollectionAttr : MAttr<"DeviceSpecCollection", "device_spec_collection"> {
  let summary = "A collection of device specifications.";
  let description = [{
    The `#M.device_spec_collection` attribute describes a collection of
    `#M.device_spec` attributes along with the device reference for the
    unique 'host' device. Each `#M.device_spec` must have a distinct device
    reference, and obviously the 'host' device reference must exist.

    ```mlir
    #M.device_spec_collection<
      host = <"cpu", 0>,
      devices = [<ref = <"cpu", 0>,
                  target = <triple="x86_64-unknown-linux-gnu", arch="znver3",
                            features="+avx2">>,
                 <ref = <"gpu", 1>,
                  target = <triple="nvptx64-nvidia-cuda", arch="sm_80">>]>
    ```

    Encoded into MEF as JSON in string form.

    Can be represented at runtime by M::DeviceSpecCollection.
  }];

  let parameters = (ins
    M_DeviceRefAttr:$host,
    ArrayRefParameter<"DeviceSpecAttr">:$devices
  );

  let assemblyFormat = [{
    `<` `host` `=` $host `,` `devices` `=` custom<DeviceSpecAttrArray>($devices) `>`
  }];

  let extraClassDeclaration = [{
    /// Returns the device spec which matches device reference, or an error if
    /// no match.
    ErrorOr<DeviceSpecAttr> findDeviceSpec(DeviceRefAttr deviceReference) const;

    /// Returns the device spec for the host.
    DeviceSpecAttr getHostDeviceSpec() const;
  }];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// InOutSignatureAttr
//===----------------------------------------------------------------------===//

def M_InOutSignatureAttr : MAttr<"InOutSignature", "inout_sig"> {
  let summary = "Describes the in/out/mut intended semantics for "
                "pointer-like operands.";
  let description = [{
    This attribute captures the intended semantics for each pointer-like
    operand of a primitive or kernel.
     - '.': Operand is a value.
     - 'i': Operand is only read.
     - 'o': Operand is only written.
     - 'm': Operand is mutated in-place.

    May be used to distinguish buffer-like operands on generic ops during
    compilation, in which case the attribute can be erased before conversion
    to MEF etc. May also be made available at runtime to help in runtime
    assertions and safety checks.

    Encoded into MEF as a string.

    Example:
    ```mlir
    #M.inout_sig<"ii.mo">
    ```
  }];

  let parameters = (ins StringRefParameter<>:$signature);
  let assemblyFormat = "`<` $signature `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
     /// The in/out/mut semantics of a single operand.
     enum InOutSemantics : char {
       kNone = '.',
       kIn = 'i',
       kOut = 'o',
       kMut = 'm'
     };

     /// Returns the arity of the signature.
     size_t size() const { return getSignature().size(); }

     /// Returns the semantics for the i'th operand.
     InOutSemantics operator[](size_t i) const {
       const StringRef& sig = getSignature();
       assert(i < sig.size() && "operand index out of range");
       return static_cast<InOutSemantics>(sig[i]);
     }

     /// Returns a new signature attribute with the specified arguments removed.
     InOutSignatureAttr remove(const llvm::BitVector& toRemove);

     /// Append new arguments to the current signature and return the result
     /// as a new signature attribute.
     InOutSignatureAttr append(ArrayRef<InOutSemantics> newArgs);
  }];

  let builders = [
    AttrBuilder<(ins "ArrayRef<InOutSemantics>":$signature)>
  ];
}



#endif // MATTRS_TD
