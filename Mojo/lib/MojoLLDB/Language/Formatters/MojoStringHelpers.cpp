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

#include "MojoStringHelpers.h"
#include "lldb/Core/Address.h"
#include "lldb/DataFormatters/StringPrinter.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/ValueObject/ValueObject.h"
#include "llvm/Support/Endian.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

namespace M::KGEN::Mojo {

bool dumpMojoString(ValueObject &context, const MojoStringHeader &header,
                    Stream &stream, const TypeSummaryOptions &options) {
  // Top bit of `capacity` ⇒ small/inline form. See
  // `Mojo/stdlib/std/collections/string/string.mojo` for the
  // layout (fields `_ptr_or_data`, `_len_or_data`, `_capacity_or_data`;
  // inline flag = `1 << (Int.BITWIDTH - 1)`).
  if (static_cast<int64_t>(header.capacity) < 0) {
    // `length` is the low 5 bits of capacity's top byte (range 0–31). The
    // inline payload spans only the 3 × 8-byte header (24 bytes minus the
    // top flag/length byte = 23 data bytes), so in a well-formed string
    // `length` never exceeds 23. Use a 32-byte zero-initialized buffer to
    // cover the 5-bit mask's full range safely — if corrupted memory yields
    // `length` between 24 and 31, the trailing bytes read back as zero
    // rather than out-of-bounds stack memory.
    size_t length = (header.capacity >> (7 * 8)) & 31;
    uint8_t bytes[32] = {};
    llvm::support::endian::write64le(bytes + 0, header.ptrOrData);
    llvm::support::endian::write64le(bytes + 8, header.lenOrData);
    llvm::support::endian::write64le(bytes + 16, header.capacity);

    DataExtractor extractor(bytes, length, lldb::ByteOrder::eByteOrderLittle,
                            8);
    StringPrinter::ReadBufferAndDumpToStreamOptions printOpts(context);
    printOpts.SetData(std::move(extractor));
    printOpts.SetStream(&stream);
    printOpts.SetPrefixToken(nullptr);
    printOpts.SetQuote('"');
    printOpts.SetSourceSize(length);
    return StringPrinter::ReadBufferAndDumpToStream<
        StringPrinter::StringElementType::ASCII>(printOpts);
  }

  // Heap form: delegate to `StringPrinter::ReadStringAndDumpToStream`,
  // which reads the payload bytes from the target for us given an
  // `Address` + `TargetSP`.
  size_t size = header.lenOrData;
  if (size == 0) {
    stream << "\"\"";
    return true;
  }

  TargetSP target = context.GetTargetSP();
  if (!target)
    return false;

  // Cap and record truncation ourselves rather than relying on LLDB's
  // built-in max-length cap. We have to pass `size + 1` as the source
  // size (see below), and LLDB's cap would bite into the +1, so we
  // disable it with `SetIgnoreMaxLength(true)`. The matching `"..."`
  // suffix that LLDB would normally emit is reproduced below.
  bool truncated = false;
  if (options.GetCapping() == TypeSummaryCapping::eTypeSummaryCapped) {
    size_t maxSize = target->GetMaximumSizeOfStringSummary();
    if (size > maxSize) {
      size = maxSize;
      truncated = true;
    }
  }

  StringPrinter::ReadStringAndDumpToStreamOptions printOpts(context);
  printOpts.SetLocation(Address(header.ptrOrData));
  printOpts.SetTargetSP(std::move(target));
  printOpts.SetStream(&stream);
  printOpts.SetPrefixToken(nullptr);
  printOpts.SetQuote('"');
  // `ReadStringAndDumpToStream` reads ASCII via `ReadCStringFromMemory`,
  // which reserves one byte of the source size for a trailing null
  // terminator even when the memory isn't actually null-terminated, so
  // pass `size + 1` to get all `size` bytes of the length-delimited Mojo
  // String payload.
  printOpts.SetSourceSize(size + 1);
  printOpts.SetHasSourceSize(true);
  printOpts.SetIgnoreMaxLength(true);
  if (!StringPrinter::ReadStringAndDumpToStream<
          StringPrinter::StringElementType::ASCII>(printOpts))
    return false;
  if (truncated)
    stream << "...";
  return true;
}

} // namespace M::KGEN::Mojo
