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

#ifndef KGEN_LIB_MOJOLLDB_TYPESYSTEM_MOJOTYPEDATALAYOUT_H
#define KGEN_LIB_MOJOLLDB_TYPESYSTEM_MOJOTYPEDATALAYOUT_H

#include "Mojo/MojoTooling/PublicASTDecl.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-private.h"

namespace M {

namespace KGEN::LIT {
class StructDeclOp;
} // namespace KGEN::LIT

class MojoParserContext;
class MojoASTTypeRef;
class TargetInfoAttr;
} // namespace M

namespace M::KGEN::Mojo {

/// Abstraction over the memory layout of Mojo types.
///
/// It contains useful information needed to read and traverse the memory of a
/// type, including its fields if it's a struct.
class MojoTypeDataLayout {
public:
  MojoTypeDataLayout(uint64_t byteSize = 0, uint64_t alignment = 1)
      : byteSize(byteSize), alignment(alignment) {}

  /// Move-only class.
  MojoTypeDataLayout(const MojoTypeDataLayout &) = delete;
  MojoTypeDataLayout &operator=(const MojoTypeDataLayout &) = delete;
  MojoTypeDataLayout(MojoTypeDataLayout &&) = default;

  /// If the owning type is a struct, this will contain the memory layout of its
  /// child fields.
  class Field {
  public:
    Field(uint64_t byteOffset, uint64_t byteSize, uint64_t alignment,
          MojoASTTypeRef concreteType)
        : byteOffset(byteOffset), byteSize(byteSize), alignment(alignment),
          concreteType(concreteType) {}

    /// Return the byte size of this field.
    uint64_t getByteSize() const { return byteSize; }

    /// Return the alignment of this field.
    uint64_t getAlignment() const { return alignment; }

    /// Return the byte offset of this field.
    uint64_t getByteOffset() const { return byteOffset; }

    /// Return the concrete type of this field gotten by resolving its
    /// parameters.
    MojoASTTypeRef getConcreteType() const { return concreteType; }

  private:
    uint64_t byteOffset;
    uint64_t byteSize;
    uint64_t alignment;
    MojoASTTypeRef concreteType;
  };

  /// Return the layout entries of each field if it's a struct, otherwise return
  /// an empty list.
  ArrayRef<Field> getFields() const { return fields; }

  /// Return the byte size of this type. It handles correctly alignment of
  /// structs.
  uint64_t getByteSize() const { return byteSize; }

  /// Return the alignment of this type.
  uint64_t getAlignment() const { return alignment; }

  /// Add a field to this type.
  void addField(const Field &field) { fields.push_back(field); }

  /// Set the byte size of this type.
  void setByteSize(uint64_t byteSize) { this->byteSize = byteSize; }

  /// Set the alignment of this type.
  void setAlignment(uint64_t alignment) { this->alignment = alignment; }

private:
  std::vector<Field> fields;
  uint64_t byteSize;
  uint64_t alignment;
};

/// Class used to query and cache the data layout of Mojo types.
///
/// Note: calculating the data layout of structs requires traversing it
/// recursively, hence the importance of using a cache to avoid recomputation of
/// nested structs.
class MojoTypeDataLayoutContext {
public:
  MojoTypeDataLayoutContext(MojoParserContext &context,
                            TargetInfoAttr targetInfo);
  ~MojoTypeDataLayoutContext();

  /// Get of calculate the data layout of the given type. If it's impossible to
  /// calculate the layout of the type or of any of its nested types in the case
  /// of a struct, then this returns null.
  const std::optional<MojoTypeDataLayout> &
  getOrCalculate(MojoASTTypeRef typeRef);

  /// Invalidate the cached layout stored for the given type. This is not
  /// recursive, i.e. invalidating a struct type doesn't automatically
  /// invalidate its members.
  void invalidateCache(MojoASTTypeRef typeRef);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_TYPESYSTEM_MOJOTYPEDATALAYOUT_H
