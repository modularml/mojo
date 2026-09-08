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

#include "MojoLanguage.h"
#include "../Utils/Errors.h"
#include "../Utils/ValueObjectHelpers.h"
#include "Formatters/MojoDecoratorBasedTypeFormatter.h"
#include "Formatters/MojoDictTypeFormatter.h"
#include "Formatters/MojoKGENVariantTypeFormatter.h"
#include "Formatters/MojoListTypeFormatter.h"
#include "Formatters/MojoPythonObjectFormatter.h"
#include "Formatters/MojoStringHelpers.h"
#include "Formatters/MojoVariantTypeFormatter.h"
#include "lldb/API/SBValue.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/DataFormatters/FormatManager.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/DataFormatters/VectorType.h"
#include <cinttypes>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace M::KGEN::Mojo;

LLDB_PLUGIN_DEFINE(MojoLanguage)

void MojoLanguage::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), "Mojo Language",
                                CreateInstance);
}

void MojoLanguage::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

//===----------------------------------------------------------------------===//
// Static Functions
//===----------------------------------------------------------------------===//

Language *MojoLanguage::CreateInstance(lldb::LanguageType language) {
  switch (language) {
  case lldb::eLanguageTypeMojo:
    return new MojoLanguage();
  default:
    return nullptr;
  }
}

static bool
builtinStringSummaryProvider(ValueObject &valobj, Stream &stream,
                             const TypeSummaryOptions &summaryOptions) {
  // If we fail to read the string, we show some placeholder error text.
  // Otherwise, if we return false, for example, LLDB would print the contents
  // of the inner List.
  auto onError = [&stream](llvm::StringRef why = {}) {
    stream << "Summary Unavailable" << why;
    return true;
  };

  auto loadWord = [&](const char *memberName, uint64_t &result) -> bool {
    // Get the _StringCapacityField member.
    ValueObjectSP field = valobj.GetChildMemberWithName(memberName);
    if (!field || !field->GetError().Success())
      return onError("; could not find String." + std::string(memberName) +
                     " field");

    field = unwrapToScalarOrPointer(field);

    if (!field || !field->GetError().Success())
      return onError("decoding " + std::string(memberName));

    if (field->IsPointerType()) {
      result = field->GetPointerValue().address;
      if (result == LLDB_INVALID_ADDRESS)
        return onError("failed to load pointer value");
      return false;
    }

    bool success = true;
    result = field->GetValueAsUnsigned(0, &success);
    if (!success)
      return onError("loading " + std::string(memberName));
    return false;
  };

  MojoStringHeader header{};
  if (loadWord("_ptr_or_data", header.ptrOrData) ||
      loadWord("_len_or_data", header.lenOrData) ||
      loadWord("_capacity_or_data", header.capacity))
    return true;

  if (!dumpMojoString(valobj, header, stream, summaryOptions))
    return onError();
  return true;
}

/// The None type is rendered nicely if its summary is the "None" string.
static bool kgenNoneSummaryProvider(ValueObject &valobj, Stream &stream,
                                    const TypeSummaryOptions &summaryOptions) {
  stream << "None";
  return true;
}

/// Bool (a !kgen.scalar<bool>) types are rendered nicely as True or False.
static bool boolSummaryProvider(ValueObject &valobj, Stream &stream,
                                const TypeSummaryOptions &summaryOptions) {
  ValueObjectSP dataVal = valobj.GetChildAtIndex(0);

  bool success = false;
  int val = dataVal->GetValueAsUnsigned(/*default=*/0, &success);
  if (success) {
    if (val == 0)
      stream << "False";
    else
      stream << "True";
    return true;
  }
  return false;
}

/// Create a short summary for a vector-like object. It includes the size of the
/// container and, if the children are scalars or have summaries of their own,
/// they will be displayed as well.
static bool
vectorLikeSummaryProvider(ValueObject &valobj, Stream &stream,
                          const TypeSummaryOptions &summaryOptions) {
  auto numChildren = getExpectedValueOr(valobj.GetNumChildren(), 0u);
  stream.Format("(size {0})", numChildren);

  // We'll limit the amount of characters to use when displaying children.
  // In practice we can go beyond this limit by a few characters.
  const size_t maxChildrenSummaryLength = 32;
  std::string childrenSummary = "[";

  size_t i = 0;
  for (; i < numChildren; ++i) {
    // If we exceeded the number of characters, we break;
    if (childrenSummary.size() > maxChildrenSummaryLength)
      break;

    ValueObjectSP child = valobj.GetChildAtIndex(i);

    llvm::StringRef childText;
    std::string childSummary;
    child->GetSummaryAsCString(childSummary, summaryOptions);
    if (childSummary.empty())
      childText = child->GetValueAsCString();
    else
      childText = childSummary;

    // If we can't generate some text for the current child, we stop.
    if (childText.empty())
      break;

    if (i > 0)
      childrenSummary += ", ";
    childrenSummary += childText;
  }

  // If we printed some children, we include them in the output stream.
  if (i > 0) {
    // If we stopped early, we add `...` to show that there are more elements.
    if (i < numChildren)
      childrenSummary += ", ...";

    childrenSummary += "]";
    stream << childrenSummary;
  }
  return true;
}

/// Format SIMD boolean vectors nicely, handling bit-packed values
static bool
simdBoolVectorSummaryProvider(ValueObject &valobj, Stream &stream,
                              const TypeSummaryOptions &summaryOptions) {
  ValueObjectSP dataVal = valobj.GetChildAtIndex(0);
  bool success = false;
  uint64_t packedData = dataVal ? dataVal->GetValueAsUnsigned(0, &success) : 0;

  if (!dataVal || !success) {
    stream.PutCString("<error>");
    return true;
  }

  stream.PutCString("{\n");
  auto numChildren = getExpectedValueOr(valobj.GetNumChildren(), 0u);
  for (size_t i = 0; i < numChildren; ++i) {
    const bool bit = (packedData >> i) & 1;
    stream.Printf("  (bool) [%zu] = %s\n", i, bit ? "True" : "False");
  }
  stream.PutCString("}");
  return true;
}

/// Summary provider for single-element scalar types (!kgen.scalar<*>).
/// Instead of displaying `([0] = 12)`, shows just `12`.
static bool scalarSummaryProvider(ValueObject &valobj, Stream &stream,
                                  const TypeSummaryOptions &summaryOptions) {
  auto numChildren = getExpectedValueOr(valobj.GetNumChildren(), 0u);
  if (numChildren != 1)
    return false;
  ValueObjectSP child = valobj.GetChildAtIndex(0);
  if (!child)
    return false;
  const char *value = child->GetValueAsCString();
  if (!value)
    return false;
  stream << value;
  return true;
}

namespace {
/// Synthetic children provider that elides `_mlir_value` wrapper fields.
/// When a `!lit.struct` type has an `_mlir_value` field that itself contains
/// more than one child (i.e. a composite MLIR type like a multi-element struct
/// or pack), this presents those children directly, removing the wrapper from
/// display. The child-count guard prevents elision for scalar-typed
/// `_mlir_value` fields (e.g. the underlying index in Int), which should
/// display their value rather than an empty child list.
class MlirValueElisionFrontEnd : public SyntheticChildrenFrontEnd {
  ValueObjectSP m_inner;

public:
  MlirValueElisionFrontEnd(ValueObject &backend)
      : SyntheticChildrenFrontEnd(backend) {}

  llvm::Expected<uint32_t> CalculateNumChildren() override {
    if (m_inner)
      return m_inner->GetNumChildren();
    return m_backend.GetNumChildren();
  }

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override {
    if (m_inner)
      return m_inner->GetChildAtIndex(idx);
    return m_backend.GetChildAtIndex(idx);
  }

  bool MightHaveChildren() override { return true; }

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override {
    if (m_inner)
      return m_inner->GetIndexOfChildWithName(name);
    return m_backend.GetIndexOfChildWithName(name);
  }

  lldb::ChildCacheState Update() override {
    static const ConstString kMlirValue("_mlir_value");
    m_inner = nullptr;
    // Use GetChildMemberWithName to find _mlir_value by struct-decl field
    // name. This is necessary for types like Tuple[T...] whose backing KGEN
    // struct type system presents the pack elements as numbered children
    // ([0], [1], …) rather than the Mojo-level _mlir_value wrapper.
    //
    // Guard: only elide when _mlir_value has more than one child. This
    // prevents applying elision to types like Int, where GetChildMemberWithName
    // finds _mlir_value recursively through an intermediate "value" field
    // (Int.value → Scalar._mlir_value = !kgen.simd<1, index>, 1 child).
    // For those types, elision would hide the scalar value rather than
    // simplifying display.
    //
    // Safety: this synthetic is registered on `^!lit\.struct<.*>` with the
    // lowest priority.  Other !lit.struct types that have a multi-child
    // _mlir_value (SIMD[T, N>1] with !kgen.simd, StaticTuple with !pop.array)
    // are decorated with @lldb_formatter_wrapping_type, so the higher-priority
    // wrapping-type synthetic takes precedence and this elision never fires
    // for them.  _RegisterPackType has !kgen.struct (like Tuple) and is
    // internal — elision is harmless there.
    //
    // Known gap: single-element Tuple[T] also has a 1-child _mlir_value
    // (!kgen.struct<T> isParamPack>), so elision does not fire and the expanded
    // children view shows `_mlir_value = ([0] = v)` rather than `[0] = v`
    // directly.  The one-liner summary is still correct because
    // packTypeSummaryProvider formats _mlir_value via the wrapping-type summary
    // path.
    auto child = m_backend.GetChildMemberWithName(kMlirValue);
    if (child && getExpectedValueOr(child->GetNumChildren(), 0u) > 1)
      m_inner = child;
    return lldb::ChildCacheState::eRefetch;
  }
};
} // namespace

static SyntheticChildrenFrontEnd *
mlirValueElisionFrontEndCreator(CXXSyntheticChildren *,
                                const ValueObjectSP &valobjSP) {
  return new MlirValueElisionFrontEnd(*valobjSP);
}

/// Summary provider for StringSpan, StaticString, and their underlying
/// Span[Byte, ...] types. Two cases:
///   1. Struct field: LLDB type is StringSpan, which has `_slice: Span[Byte]`.
///      Navigate via _slice._data and _slice._len.
///   2. Top-level variable: KGEN flattens TrivialRegisterPassable types in
///      DWARF, so the LLDB type is Span[Byte] directly with _data and _len.
static bool
stringSliceSummaryProvider(ValueObject &valobj, Stream &stream,
                           const TypeSummaryOptions &summaryOptions) {
  auto onError = [&stream](llvm::StringRef why = {}) {
    stream << "Summary Unavailable" << why;
    return true;
  };

  // Case 1: valobj is StringSpan — navigate to _slice (the Span[Byte] field).
  // Case 2: valobj is already the Span[Byte] (flattened top-level variable).
  //
  // We detect the case via the type name rather than probing for _slice,
  // because calling GetChildMemberWithName on a synthetic ValueObject for a
  // non-existent child triggers a cantFail abort inside LLDB.
  //
  // This function is registered for two type patterns (see
  // LoadLibMojoFormatters):
  //   - StringSpan.*  → type name contains "StringSpan" → case 1
  //   - Span[Byte]     → type name does not              → case 2
  // The type name check here must remain consistent with those registrations.
  ValueObjectSP span;
  llvm::StringRef typeName = valobj.GetTypeName().GetStringRef();
  if (typeName.contains("StringSpan")) {
    ValueObjectSP slice = valobj.GetChildMemberWithName("_slice");
    if (!slice || !slice->GetError().Success())
      return onError("; could not find _slice field");
    span = slice;
  } else {
    span = valobj.GetSP();
  }

  // Get the length from _len (an Int) and data pointer from _data
  // (UnsafePointer[Byte]). Both may be wrapped in Mojo value-type layers
  // (_mlir_value, value, address, etc.) — unwrap to reach the scalar/pointer.
  ValueObjectSP lenField =
      unwrapToScalarOrPointer(span->GetChildMemberWithName("_len"));
  if (!lenField || !lenField->GetError().Success() || !lenField->IsScalarType())
    return onError("; could not find _len");

  bool success = false;
  size_t size = lenField->GetValueAsUnsigned(0, &success);
  if (!success)
    return onError("; could not read _len");

  // If the size is 0, the data address might be invalid.
  if (size == 0) {
    stream << "\"\"";
    return true;
  }

  ValueObjectSP dataPointer =
      unwrapToScalarOrPointer(span->GetChildMemberWithName("_data"));
  if (!dataPointer || !dataPointer->GetError().Success() ||
      !dataPointer->IsPointerType())
    return onError("; could not find data pointer in _data");

  StringPrinter::ReadBufferAndDumpToStreamOptions options(valobj);
  if (summaryOptions.GetCapping() == TypeSummaryCapping::eTypeSummaryCapped) {
    size_t maxSize = valobj.GetTargetSP()->GetMaximumSizeOfStringSummary();
    if (size > maxSize) {
      size = maxSize;
      options.SetIsTruncated(true);
    }
  }

  DataExtractor extractor;
  const size_t bytesRead = dataPointer->GetPointeeData(extractor, 0, size);
  if (bytesRead < size)
    return onError("; couldn't fetch string data");

  options.SetData(std::move(extractor));
  options.SetStream(&stream);
  options.SetPrefixToken(nullptr);
  options.SetQuote('"');
  options.SetSourceSize(size);
  return StringPrinter::ReadBufferAndDumpToStream<
      StringPrinter::StringElementType::ASCII>(options);
}

// Summary provider for Mojo UnsafePointer[T] variables (!kgen.pointer<T>).
// Dereferences the pointer and shows the pointee's own summary (or raw value).
// The unanchored regex `pointer<.*>` is a substring match that catches
// `!kgen.pointer<T>` type spellings.
//
// For pointer<…String…> spellings, reverse-order registration routes the
// variable to pointerToStringSummaryProvider first (that provider is
// registered later). For other pointer-to-String cases the deref path here
// reaches the String struct's own summary provider directly.
static bool
mojoPointerSummaryProvider(ValueObject &valobj, Stream &stream,
                           const TypeSummaryOptions &summaryOptions) {
  if (!valobj.IsPointerType())
    return false;

  Status error;
  ValueObjectSP deref = valobj.Dereference(error);
  if (!deref || error.Fail())
    return false;

  // Avoid recursing into pointer-to-pointer.
  if (deref->GetCompilerType().IsPointerType())
    return false;

  // Call GetValueAsCString before GetSummaryAsCString to prime the data
  // buffer. GetSummaryAsCString triggers UpdateValueIfNeeded on a freshly
  // Dereference()'d value whose m_data is still null; UpdateValue() returns
  // true (no error path), so the caller immediately calls Checksum(null, 0),
  // which fires a UBSAN nonnull violation in LLVM's MD5 code.
  // GetValueAsCString also calls UpdateValueIfNeeded and loads data into
  // m_data, so after this call NeedsUpdating() is false and the
  // GetSummaryAsCString below skips Checksum entirely.
  //
  // For vector/SIMD types (e.g. Float64 = SIMD[DType.float64, 1]), the raw
  // value is hex bytes — not useful. Fall through to the registered summary
  // provider even if GetValueAsCString returns non-null.
  const char *raw = deref->GetValueAsCString();
  bool isVector =
      (deref->GetCompilerType().GetTypeInfo() & lldb::eTypeIsVector) != 0;
  if (raw && !isVector) {
    stream << raw;
    return true;
  }

  // For types with no raw scalar LLDB value (e.g. Bool), let the pointee's
  // own registered formatter produce the summary.
  std::string summary;
  deref->GetSummaryAsCString(summary, summaryOptions);
  if (!summary.empty()) {
    stream << summary;
    return true;
  }

  return false;
}

// Summary provider for pointer<String> types (including UnsafePointer[String])
static bool
pointerToStringSummaryProvider(ValueObject &valobj, Stream &stream,
                               const TypeSummaryOptions &summaryOptions) {
  // Get the type name
  ConstString typeNameConst = valobj.GetTypeName();
  if (!typeNameConst)
    return false;

  // Check if this is a pointer to a String type
  llvm::StringRef typeName = typeNameConst.GetStringRef();
  if (!typeName.contains("String"))
    return false;

  // Only handle pointer types
  if (!valobj.IsPointerType())
    return false;

  // Try to dereference the pointer to get the String object
  Status error;
  ValueObjectSP derefValObj = valobj.Dereference(error);
  if (!derefValObj || error.Fail())
    return false;

  // Check if the dereferenced type is a String
  ConstString derefTypeName = derefValObj->GetTypeName();
  if (!derefTypeName)
    return false;

  llvm::StringRef derefType = derefTypeName.GetStringRef();
  if (!derefType.contains("String"))
    return false;

  // The dereferenced object is a String, so we need to format it
  // Use the existing string formatter
  return builtinStringSummaryProvider(*derefValObj, stream, summaryOptions);
}

static void
LoadLibMojoFormatters(const lldb::TypeCategoryImplSP &mojoCategorySP) {
  if (!mojoCategorySP)
    return;

  // These settings are the same as the C++ ones.
  SyntheticChildren::Flags synthFlags;
  synthFlags.SetCascades(true).SetSkipPointers(true).SetSkipReferences(true);

  constexpr const char *kListRegex =
      R"(^!lit.struct<@std::@collections::@list::@"?List[\[<].*)";
  constexpr const char *kDictRegex =
      R"(^!lit.struct<@std::@collections::@dict::@"?Dict[\[<].*)";
  constexpr const char *kLLDBFormatterWrappingTypeRegex =
      R"(.* {@std::utils::_visualizers::lldb_formatter_wrapping_type\(.*)";
  // Matches both DWARF form (@"Variant[...]) and REPL form (@Variant<...>).
  constexpr const char *kVariantRegex =
      R"(^!lit\.struct<@std::@utils::@variant::@"?Variant[\[<].*)";
  // Matches both DWARF form (@"Optional[...]) and REPL form (@Optional<...>).
  constexpr const char *kOptionalRegex =
      R"(^!lit\.struct<@std::@collections::@optional::@"?Optional[\[<].*)";

  // Formatters are matched in reverse order (last registered = highest
  // priority). The _mlir_value elision is registered first so it has the
  // lowest priority; more specific formatters take precedence.
  AddCXXSynthetic(mojoCategorySP, mlirValueElisionFrontEndCreator,
                  "Mojo _mlir_value elision", R"(^!lit\.struct<.*>)",
                  synthFlags, /*regex=*/true);
  AddCXXSynthetic(mojoCategorySP,
                  MojoLLDBWrappingTypeTypeSyntheticFrontEndCreator,
                  "Mojo decorator-based synthetic children",
                  kLLDBFormatterWrappingTypeRegex, synthFlags,
                  /*regex=*/true);
  AddCXXSynthetic(mojoCategorySP, mojoListSyntheticFrontEndCreator,
                  "Mojo List synthetic children", kListRegex, synthFlags,
                  /*regex=*/true);
  AddCXXSynthetic(mojoCategorySP, mojoKGENVariantSyntheticFrontEndCreator,
                  "Mojo !kgen.variant synthetic children",
                  R"(^!kgen\.variant<.*>)", synthFlags, /*regex=*/true);
  AddCXXSynthetic(mojoCategorySP, mojoDictSyntheticFrontEndCreator,
                  "Mojo Dict synthetic children", kDictRegex, synthFlags,
                  /*regex=*/true);

  TypeSummaryImpl::Flags summaryFlags;
  summaryFlags.SetCascades(true)
      .SetSkipPointers(false)
      .SetSkipReferences(false)
      .SetDontShowChildren(false)
      .SetDontShowValue(true)
      .SetShowMembersOneLiner(false)
      .SetHideItemNames(false);

  // Summary providers are matched in reverse order.
  AddCXXSummary(mojoCategorySP, MojoLLDBWrappingTypeSummaryProvider,
                "Mojo decorator-based summary provider",
                kLLDBFormatterWrappingTypeRegex, summaryFlags, /*regex=*/true);

  AddCXXSummary(mojoCategorySP, mojoPointerSummaryProvider,
                "Mojo UnsafePointer summary provider", R"(pointer<.*>)",
                summaryFlags, /*regex=*/true);

  // Add summary provider for pointer<String> types - MUST be AFTER generic
  // pointer handler because summary providers are matched in reverse order
  AddCXXSummary(mojoCategorySP, pointerToStringSummaryProvider,
                "pointer<String> summary provider",
                R"(pointer<(@std::)?@collections::@string::@string::@String>)",
                summaryFlags, /*regex=*/true);

  summaryFlags.SetDontShowChildren(true);
  AddCXXSummary(mojoCategorySP, kgenNoneSummaryProvider,
                "!kgen.none summary provider", "!kgen.none", summaryFlags,
                /*regex=*/false);

  AddCXXSummary(mojoCategorySP, simdBoolVectorSummaryProvider,
                "SIMD bool vector summary provider", "!kgen.simd<[0-9]+, bool>",
                summaryFlags, /*regex=*/true);

  AddCXXSummary(mojoCategorySP, boolSummaryProvider, "bool summary provider",
                "!kgen.scalar<bool>", summaryFlags, /*regex=*/false);

  AddCXXSummary(mojoCategorySP, scalarSummaryProvider,
                "scalar summary provider", R"(!kgen\.scalar<[^>]+>)",
                summaryFlags, /*regex=*/true);

  AddCXXSummary(
      mojoCategorySP, builtinStringSummaryProvider,
      "collections::string::string::String summary provider",
      R"(!lit.struct<(@std::)?@collections::@string::@string::@String>)",
      summaryFlags, /*regex=*/true);

  // Must be registered AFTER String (reverse-order matching means this takes
  // priority for StringSpan types, which have a distinct type name).
  AddCXXSummary(
      mojoCategorySP, stringSliceSummaryProvider,
      "collections::string::string_span::StringSpan summary provider",
      R"(!lit.struct<(@std::)?@collections::@string::@string_span::@StringSpan.*>)",
      summaryFlags, /*regex=*/true);

  // TrivialRegisterPassable types like StringSpan and StaticString are
  // flattened to their inner Span[Byte] type in DWARF for top-level variables.
  // Register a formatter for Span[Byte, ...] (element type ui8) so these
  // variables get summaries. Restricted to ui8 to avoid matching non-byte
  // spans. The two regexes (StringSpan and Span[Byte]) are non-overlapping by
  // construction, so registration order only matters relative to other
  // formatters. Registered last so it is tried first in the reverse list.
  AddCXXSummary(
      mojoCategorySP, stringSliceSummaryProvider,
      "memory::span::Span[Byte] summary provider (flattened StringSpan)",
      R"(!lit.struct<(@std::)?@collections::@span::@.Span\[.*ui8.*>)",
      summaryFlags,
      /*regex=*/true);

  summaryFlags.SetDontShowChildren(false);
  summaryFlags.SetDontShowValue(false);
  AddCXXSummary(mojoCategorySP, vectorLikeSummaryProvider,
                "collections::list::List summary provider", kListRegex,
                summaryFlags, /*regex=*/true);
  AddCXXSummary(mojoCategorySP, mojoVariantSummaryProvider,
                "Variant summary provider", kVariantRegex, summaryFlags,
                /*regex=*/true);
  AddCXXSummary(mojoCategorySP, mojoOptionalSummaryProvider,
                "Optional summary provider", kOptionalRegex, summaryFlags,
                /*regex=*/true);
  AddCXXSummary(mojoCategorySP, vectorLikeSummaryProvider,
                "collections::dict::Dict summary provider", kDictRegex,
                summaryFlags, /*regex=*/true);

  summaryFlags.SetDontShowChildren(true);
  AddCXXSummary(
      mojoCategorySP, mojoPythonObjectSummaryProvider,
      "Mojo PythonObject summary provider",
      R"(!lit\.struct<(@std::)?@python::@python_object::@PythonObject>)",
      summaryFlags, /*regex=*/true);
}

lldb::TypeCategoryImplSP MojoLanguage::GetFormatters() {
  static llvm::once_flag initialize;
  static TypeCategoryImplSP category;

  llvm::call_once(initialize, [this]() -> void {
    DataVisualization::Categories::GetCategory(ConstString(GetPluginName()),
                                               category);
    if (category) {
      LoadLibMojoFormatters(category);
    }
  });
  return category;
}

HardcodedFormatters::HardcodedSummaryFinder
MojoLanguage::GetHardcodedSummaries() {
  return HardcodedFormatters::HardcodedSummaryFinder();
}

HardcodedFormatters::HardcodedSyntheticFinder
MojoLanguage::GetHardcodedSynthetics() {
  return HardcodedFormatters::HardcodedSyntheticFinder();
}
