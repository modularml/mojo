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

#ifndef KGEN_LIB_MOJOLLDB_LANGUAGE_MOJOKGENVARIANTTYPEFORMATTER_H
#define KGEN_LIB_MOJOLLDB_LANGUAGE_MOJOKGENVARIANTTYPEFORMATTER_H

#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/lldb-forward.h"

namespace M::KGEN::Mojo {
class MojoKGENVariantTypeSyntheticFrontEnd
    : public lldb_private::SyntheticChildrenFrontEnd {
public:
  MojoKGENVariantTypeSyntheticFrontEnd(const lldb::ValueObjectSP &backend);

  ~MojoKGENVariantTypeSyntheticFrontEnd() override = default;

  llvm::Expected<uint32_t> CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;

  lldb::ChildCacheState Update() override;

  bool MightHaveChildren() override;

  llvm::Expected<size_t>
  GetIndexOfChildWithName(lldb_private::ConstString name) override;

  /// Parse the given `ValueObject` representing a kgen.variant.
  /// Returns the ValueObjectSP for the active variant.
  static std::optional<lldb::ValueObjectSP>
  parseKGENVariant(lldb::ValueObjectSP valobj);

private:
  lldb::ValueObjectSP content;
};

lldb_private::SyntheticChildrenFrontEnd *
mojoKGENVariantSyntheticFrontEndCreator(lldb_private::CXXSyntheticChildren *,
                                        const lldb::ValueObjectSP &valobjSP);
} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_LANGUAGE_MOJOKGENVARIANTTYPEFORMATTER_H
