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

#ifndef KGEN_LIB_MOJOLLDB_LANGUAGE_MOJODECORATORBASEDTYPEFORMATTER_H
#define KGEN_LIB_MOJOLLDB_LANGUAGE_MOJODECORATORBASEDTYPEFORMATTER_H

#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/lldb-forward.h"

namespace M::KGEN::Mojo {
/// Synthetic type factory that handles formatters specified via mojo
/// decorators.
lldb_private::SyntheticChildrenFrontEnd *
MojoLLDBWrappingTypeTypeSyntheticFrontEndCreator(
    lldb_private::CXXSyntheticChildren *, const lldb::ValueObjectSP &valobjSP);

/// Summary provider that handles synthetic types handled by
/// MojoLLDBWrappingTypeTypeSyntheticFrontEndCreator.
bool MojoLLDBWrappingTypeSummaryProvider(
    lldb_private::ValueObject &valobj, lldb_private::Stream &stream,
    lldb_private::TypeSummaryOptions summaryOptions);
} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_LANGUAGE_MOJODECORATORBASEDTYPEFORMATTER_H
