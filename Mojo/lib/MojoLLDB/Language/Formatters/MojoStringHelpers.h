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

#ifndef KGEN_LIB_MOJOLLDB_LANGUAGE_FORMATTERS_MOJOSTRINGHELPERS_H
#define KGEN_LIB_MOJOLLDB_LANGUAGE_FORMATTERS_MOJOSTRINGHELPERS_H

#include "lldb/lldb-forward.h"
#include <cstdint>

namespace lldb_private {
class Stream;
class TypeSummaryOptions;
class ValueObject;
} // namespace lldb_private

namespace M::KGEN::Mojo {

/// The three raw header words of a Mojo `String`, pre-decoded by the
/// caller. Field names mirror the stdlib (`Mojo/stdlib/std/
/// collections/string/string.mojo`): the `_ptr_or_data`/`_len_or_data`/
/// `_capacity_or_data` union repurposes these words for either the
/// inline (small-string) form or the heap (large-string) form.
struct MojoStringHeader {
  uint64_t ptrOrData;
  uint64_t lenOrData;
  uint64_t capacity;
};

/// Format a Mojo `String` given its pre-decoded three-word header.
///
/// Shared between the stdlib `String` summary (which reads the three
/// words via named fields on `!lit.struct<...String>`) and the Variant
/// formatter (which reads them positionally on the lowered
/// `!kgen.struct<(pointer, index, index) memoryOnly>` inside a
/// `!pop.union`). The inline/heap split and small-string byte repacking
/// live here so the two callers stay in sync; the heap path delegates to
/// `StringPrinter::ReadStringAndDumpToStream`, which reads from the
/// target directly using `context`'s `TargetSP`.
///
/// Returns false on any failure; callers typically fall back to printing
/// just the type name.
bool dumpMojoString(lldb_private::ValueObject &context,
                    const MojoStringHeader &header,
                    lldb_private::Stream &stream,
                    const lldb_private::TypeSummaryOptions &options);

} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_LANGUAGE_FORMATTERS_MOJOSTRINGHELPERS_H
