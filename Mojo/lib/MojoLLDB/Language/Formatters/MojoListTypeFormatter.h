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

#ifndef KGEN_LIB_MOJOLLDB_LANGUAGE_MOJOLISTTYPEFORMATTER_H
#define KGEN_LIB_MOJOLLDB_LANGUAGE_MOJOLISTTYPEFORMATTER_H

#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/lldb-forward.h"

namespace M::KGEN::Mojo {
class MojoListSyntheticFrontEnd
    : public lldb_private::SyntheticChildrenFrontEnd {
public:
  MojoListSyntheticFrontEnd(const lldb::ValueObjectSP &backend);

  ~MojoListSyntheticFrontEnd() override = default;

  llvm::Expected<uint32_t> CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;

  lldb::ChildCacheState Update() override;

  bool MightHaveChildren() override;

  llvm::Expected<size_t>
  GetIndexOfChildWithName(lldb_private::ConstString name) override;

  /// Parse the given `ValueObject` representing a List.
  ///
  /// Return a pair `<data pointer, size>`, where `data pointer` represents
  /// the start of the underlying data, and `size` represents the number of
  /// entries. If `size` is 0, then the data pointer might point to an invalid
  /// address.
  /// Otherwise, if `size` is larger than 0, the data pointer points to some
  /// address.
  ///
  /// This function returns null if it was not possible to read some of these
  /// fields, or if the invariants mentioned above don't hold.
  static std::optional<std::pair<lldb::ValueObjectSP, size_t>>
  parseList(lldb::ValueObjectSP valobj);

private:
  lldb::addr_t start;
  size_t size;
  lldb_private::CompilerType elementType;
  uint64_t elementSize;
};

lldb_private::SyntheticChildrenFrontEnd *
mojoListSyntheticFrontEndCreator(lldb_private::CXXSyntheticChildren *,
                                 const lldb::ValueObjectSP &valobjSP);
} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_LANGUAGE_MOJOLISTYPEFORMATTER_H
