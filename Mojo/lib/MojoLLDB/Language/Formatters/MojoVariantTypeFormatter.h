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

#ifndef KGEN_LIB_MOJOLLDB_LANGUAGE_MOJOVARIANTTYPEFORMATTER_H
#define KGEN_LIB_MOJOLLDB_LANGUAGE_MOJOVARIANTTYPEFORMATTER_H

#include "lldb/lldb-forward.h"

namespace lldb_private {
class Stream;
class TypeSummaryOptions;
class ValueObject;
} // namespace lldb_private

namespace M::KGEN::Mojo {

bool mojoVariantSummaryProvider(lldb_private::ValueObject &valobj,
                                lldb_private::Stream &stream,
                                lldb_private::TypeSummaryOptions options);

// Declared here (rather than a separate MojoOptionalTypeFormatter.h) because
// the implementation shares file-local helpers in MojoVariantTypeFormatter.cpp
// (parseVariantInfo, appendPayloadIfPresent) that are not externally visible.
bool mojoOptionalSummaryProvider(lldb_private::ValueObject &valobj,
                                 lldb_private::Stream &stream,
                                 lldb_private::TypeSummaryOptions options);

} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_LANGUAGE_MOJOVARIANTTYPEFORMATTER_H
