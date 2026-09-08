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

#ifndef INTERPRETERATTRS_TD
#define INTERPRETERATTRS_TD

include "Mojo/Interpreter/InterpreterDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

class InterpreterAttr<string attrName, string attrMnemonic,
                      list<Trait> traits = []>
    : AttrDef<InterpreterDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// PointerAttr
//===----------------------------------------------------------------------===//

def PointerAttr : InterpreterAttr<"Pointer", "pointer",
    [TypedAttrInterface]> {
  let summary = "an opaque pointer attribute";
  let description = [{
    The `#interp.pointer` attribute represents an opaque pointer value as an
    address. Pointer constants have no meaning outside of in-memory IR, so the
    textual format of this attribute has no inherent meaning. It is meant to be
    used in operation folders or the interpreter.
  }];

  let parameters = (ins "int64_t":$addr, AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "`<` $addr `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "int64_t":$addr, "Type":$type), [{
      return get(type.getContext(), addr, type);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// StoreToMemAttr
//===----------------------------------------------------------------------===//

def StoreToMemAttr : InterpreterAttr<"StoreToMem", "store_to_mem",
    [TypedAttrInterface]> {
  let summary = "wrap an attribute in stack memory";
  let description = [{
    The `#interp.store_to_mem` attribute indicates to the interpreter that the
    value should be stored into memory before used.

    Example:

    ```mlir
    #interp.store_to_mem<5 : index> : !M.pointer<index>
    ```
  }];

  let parameters = (ins "TypedAttr":$value,
                        AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "`<` $value `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "TypedAttr":$value, "Type":$type), [{
      return $_get(value.getContext(), value, type);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// AlignedBlob
//===----------------------------------------------------------------------===//

def AlignedBlob : AttrParameter<"AlignedBlob", "an aligned byte array"> {
  let allocator = [{
    {
      size_t blobSize = $_self.data.size();
      void *blobMemory = $_allocator.allocate(blobSize, $_self.align);
      memcpy(blobMemory, $_self.data.data(), blobSize);
      ArrayRef<char> data((const char *)blobMemory, blobSize);
      $_dst = AlignedBlob($_self.align, data, $_self.isString);
    }
  }];
  let cppStorageType = "OwnedAlignedBlob";
}

//===----------------------------------------------------------------------===//
// MemoryHandleAttr
//===----------------------------------------------------------------------===//

def MemoryHandleAttr : InterpreterAttr<"MemoryHandle", "memory_handle"> {
  let summary = "reference to a physical blob of memory";
  let description = [{
    The `#interp.memory_handle` attribute contains an array of aligned bytes. It
    is the backing memory of materialized interpreter memory blobs.
  }];

  let parameters = (ins AlignedBlob:$memory);
  let assemblyFormat = "`<` custom<AlignedHex>($memory) `>`";

  let builders = [
    AttrBuilder<(ins "uint64_t":$align, "ArrayRef<char>":$memory), [{
      return $_get($_ctxt, AlignedBlob{align, memory, /*isString=*/false});
    }]>,
    AttrBuilder<(ins "StringRef":$str)>
  ];

  let extraClassDeclaration = [{
    const char *getData() { return getMemory().data.data(); }
    size_t getSize() {  return getMemory().data.size(); }
    uint64_t getAlign() { return getMemory().align; }
    bool isString() { return getMemory().isString; }
  }];
}

//===----------------------------------------------------------------------===//
// CoordinateAttr
//===----------------------------------------------------------------------===//

def CoordinateAttr : InterpreterAttr<"Coordinate", "coord", [TypedAttrInterface]> {
  let summary = "represents a reference into memory";
  let description = [{
    The `#interp.coord` attribute represents a reference into memory modeled as an
    array of bytes. The first element, refSlot, is the index of row. The second
    element, offset, is the number of bytes into the row where the reference begins.
  }];
  let parameters = (ins  "uint64_t":$offset, "uint64_t":$refSlot, AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "`(` $offset `,` $refSlot `)`";
  let builders = [
      AttrBuilderWithInferredContext<(ins "int64_t":$offset, "int64_t":$refSlot,
                                          "Type":$type), [{
        return $_get(type.getContext(), offset, refSlot, type);
      }]>
  ];
}

//===----------------------------------------------------------------------===//
// MemoryBlobAttr
//===----------------------------------------------------------------------===//

def MemoryBlobAttr : InterpreterAttr<"MemoryBlob", "memory_blob"> {
  let summary = "represents a single interpreter memory blob";
  let description = [{
    The `#interp.memory_blob` attribute represents a single interpreter memory
    blob, which backs a single allocation. It contains a reference to the
    underlying resource that contains the memory value, the kind of allocation,
    and the tracked pointer regions inside the blob.
  }];

  let parameters = (ins
    "MemoryHandleAttr":$handle,
    "MemoryKind":$kind,
    ArrayRefParameter<"PointerRegion">:$pointerRegions,
    ArrayRefParameter<"int64_t">:$symbolRegions,
    DefaultValuedParameter<"unsigned", "0">:$addressSpace
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins
        "MemoryHandleAttr":$handle,
        "MemoryKind":$kind,
        "ArrayRef<PointerRegion>":$pointerRegions,
        "ArrayRef<int64_t>":$symbolRegions,
        "unsigned":$addressSpace), [{
      return $_get(handle.getContext(), handle, kind, pointerRegions,
                   symbolRegions, addressSpace);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// MemorySpaceAttr
//===----------------------------------------------------------------------===//

def MemorySpaceAttr : ArrayOfAttr<InterpreterDialect, "MemorySpace",
                                  "memory_space", "MemoryBlobAttr"> {
  let summary = "IR representation of a memory space and values";
  let description = [{
    The `#interp.memory_space` attribute describes an abstract memory space
    consisting of distinct memory allocations represented as blobs. The contents
    of the blobs can be either dumb bits or pointers. Pointers are specially
    indicated and encoded, consisting of the blob index and the offset.

    Example:

    ```mlir
    #interp.memory_space<[
      (blob0, stack, [(24, 1, 8), (50, 0, 0), (64, 1, 8))]),
      (blob1, heap, [(0, 1, 8), (8, 1, 16)])
    ]>
    ```
  }];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// SymbolArrayAttr
//===----------------------------------------------------------------------===//

def SymbolArrayAttr
    : ArrayOfAttr<InterpreterDialect, "SymbolArray", "symbols", "TypedAttr">;

//===----------------------------------------------------------------------===//
// MemoryModelAttr
//===----------------------------------------------------------------------===//

def MemoryModelAttr : InterpreterAttr<"MemoryModel", "model"> {
  let summary = "a memory table paired with a symbol table";
  let description = [{
    The `#interp.model` pairs a memory space with a symbol table.
    The symbol table contains attributes that do not have a runtime
    representation. The entries in the memory space can reference
    the symbol table and the symbols in the symbol table can reference
    the blobs of a memory space.

    `selfBlobIndex`, when set, is the blob holding the interpreter storage the
    value carrying this model was read out of. Pointers into that blob are
    references to the value's own storage: materializing the value has to point
    them at wherever the value lands, not rebuild the blob as a second object.
  }];
  let parameters = (ins "MemorySpaceAttr":$memory, "SymbolArrayAttr":$symbols,
                        OptionalParameter<"std::optional<int64_t>">:$selfBlobIndex);
  let assemblyFormat = [{
      `{` $memory `,` $symbols (`,` `self` `=` $selfBlobIndex^)? `}`
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins "MemorySpaceAttr":$memory,
                                        "SymbolArrayAttr":$symbols), [{
      return $_get(memory.getContext(), memory, symbols, std::nullopt);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// MemRefAttr
//===----------------------------------------------------------------------===//

def MemRefAttr : InterpreterAttr<"MemRef", "memref", [TypedAttrInterface]> {
  let summary = "a reference to a block of memory";
  let description = [{
    The `#interp.memref` attribute represents a reference to a block of memory. The
    block of memory lives in a dialect resource and describes the contents of
    the memory, and this attribute describes an offset into the block, which
    may be out-of-bounds, and how to interpret the memory.

    TODO: Add address space and write protection.

    Example:

    ```mlir
    #interp.memref<my_blob, 24> : !M.pointer<index>
    ```
  }];

  let parameters = (ins "MemoryModelAttr":$model,
                        "int64_t":$index,
                        "int64_t":$offset,
                        AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "`<` $model `,` $index `,` $offset `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "MemoryModelAttr":$model,
                                        "int64_t":$index, "int64_t":$offset,
                                        "Type":$type), [{
      return $_get(type.getContext(), model, index, offset, type);
    }]>
  ];

  let genVerifyDecl = 1;
}


//===----------------------------------------------------------------------===//
// UninitMemAttr
//===----------------------------------------------------------------------===//

def UninitMemAttr : InterpreterAttr<"UninitMem", "uninitmem",
                                    [TypedAttrInterface]> {
  let summary = "representation of loading uninitialized memory";
  let description = [{
    The `#interp.uninitmem` attribute represents uninitialized memory that can
    be loaded and stored but not turned into a concrete value.
  }];
  let parameters = (ins AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "";

  let builders = [TypeBuilderWithInferredContext<(ins "Type":$type), [{
    return get(type.getContext(), type);
  }]>];
}

#endif // INTERPRETERATTRS_TD
