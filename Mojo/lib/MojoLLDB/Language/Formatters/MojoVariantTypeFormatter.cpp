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

#include "MojoVariantTypeFormatter.h"
#include "../../Utils/Errors.h"
#include "../../Utils/ValueObjectHelpers.h"
#include "MojoStringHelpers.h"
#include "MojoVariantTypeNameParser.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace M::KGEN::Mojo;

namespace {

/// Fully-qualified name of the stdlib `String` struct. Must match the regex
/// used to register `builtinStringSummaryProvider` in `MojoLanguage.cpp`.
constexpr llvm::StringLiteral kStringFQN =
    "std::collections::string::string::String";

struct VariantInfo {
  size_t discriminant;
  llvm::SmallVector<VariantTypeName> typeNames;
  /// The `ValueObject` for the currently-active payload, or null if it could
  /// not be extracted. When null, the summary shows only the type name.
  ValueObjectSP activeValue;
};

/// Parse discriminant, type-name list, and active payload from a stdlib
/// `Variant` ValueObject.
///
/// Navigates `_storage._impl` where `_impl` is `!kgen.variant<T0, T1, ...>`.
/// After KGEN's LLVM lowering this is a struct with two fields:
///   [0] = `!pop.union<T0, T1, ...>` (members named `v0`, `v1`, ...)
///   [1] = `!kgen.scalar<ui8>`        (discriminant; its first child is the
///                                    raw u8)
/// The active payload is `union->GetChildAtIndex(discriminant)`.
///
/// See `Mojo/lib/Transforms/LowerCallingConventions.cpp::lowerVariantType`
/// for the `kgen.variant` → `kgen.struct<(union, scalar<ui8>)>` lowering
/// and `Mojo/lib/KGENToLLVM/DebugInfoTypeConverter.cpp::buildDebugType(
/// POP::UnionType)` for the DWARF `DW_TAG_variant_part` emission whose
/// `v0`/`v1`/... member names this formatter relies on.
///
/// In the REPL, `_impl` may be wrapped in a `!lit.struct` with `_mlir_value`;
/// the function unwraps this before reading the discriminant.
///
/// Note: in the REPL, `Variant`'s variadic type parameter causes
/// `getIfStructDecl` to fail, so `_storage` is inaccessible and this returns
/// `{}`. The summary falls back to empty in that case.
static std::optional<VariantInfo> parseVariantInfo(ValueObjectSP valobj) {
  if (!valobj || !valobj->GetError().Success())
    return {};

  valobj = valobj->GetNonSyntheticValue();
  if (!valobj)
    return {};

  ValueObjectSP storage = nonSyntheticChild(*valobj, "_storage");
  if (!storage)
    return {};

  // `_DefaultVariantStorage` has `_impl: !kgen.variant<...>`.
  // `_NichedOptionalStorage` has no `_impl`; fall through gracefully.
  ValueObjectSP impl = storage->GetChildMemberWithName("_impl");
  if (!impl || !impl->GetError().Success())
    return {};

  // In the REPL, _impl may be a !lit.struct wrapper around the underlying
  // !kgen.variant. Unwrap via _mlir_value if present.
  if (ValueObjectSP inner = impl->GetChildMemberWithName("_mlir_value"))
    if (inner->GetError().Success())
      impl = inner;

  auto numChildren = getExpectedValueOr(impl->GetNumChildren(), 0u);
  if (numChildren < 2)
    return {};

  // Discriminant is the last child: `!kgen.scalar<ui8>`.
  // `!kgen.scalar<ui8>` is not a raw scalar in LLDB — its first child IS.
  ValueObjectSP discrScalar = impl->GetChildAtIndex(numChildren - 1);
  if (!discrScalar || !discrScalar->GetError().Success())
    return {};

  ValueObjectSP discrRaw = discrScalar->GetChildAtIndex(0);
  if (!discrRaw || !discrRaw->GetError().Success())
    return {};

  bool success = false;
  size_t discr = discrRaw->GetValueAsUnsigned(0, &success);
  if (!success)
    return {};

  auto typeNames =
      extractVariantTypeNames(storage->GetTypeName().GetStringRef());
  if (typeNames.empty() || discr >= typeNames.size())
    return {};

  // The union (first child of `_impl`) has one member per Variant arm. If
  // the string-based type-name parser ever drifts from reality — as it did
  // before the aarch64-Linux fix (see MOCO-3787) — cross-check against the
  // union's arm count and bail on mismatch so we fall back cleanly instead
  // of silently rendering a wrong `displayName`.
  ValueObjectSP unionField = impl->GetChildAtIndex(0);
  if (!unionField || !unionField->GetError().Success())
    return {};
  auto unionChildren = getExpectedValueOr(unionField->GetNumChildren(), 0u);
  if (typeNames.size() != unionChildren)
    return {};

  // Extract the active payload from the union. Union members are emitted
  // as `v0`, `v1`, ..., so the discriminant is also the index.
  ValueObjectSP activeValue;
  if (ValueObjectSP candidate = unionField->GetChildAtIndex(discr))
    if (candidate->GetError().Success())
      activeValue = std::move(candidate);

  return VariantInfo{discr, std::move(typeNames), std::move(activeValue)};
}

/// Read the 3 header words of a Mojo String that lives in the lowered
/// `!kgen.struct<(pointer, index, index) memoryOnly>` shape used inside a
/// variant's union. Returns false on any access error.
static bool readLoweredStringHeader(ValueObjectSP active,
                                    MojoStringHeader &header) {
  if (!active)
    return false;
  auto numChildren = getExpectedValueOr(active->GetNumChildren(), 0u);
  if (numChildren != 3)
    return false;

  ValueObjectSP ptrField = unwrapToScalarOrPointer(active->GetChildAtIndex(0));
  ValueObjectSP lenField = active->GetChildAtIndex(1);
  ValueObjectSP capField = active->GetChildAtIndex(2);
  if (!ptrField || !lenField || !capField || !ptrField->GetError().Success() ||
      !lenField->GetError().Success() || !capField->GetError().Success())
    return false;

  bool ok = true;
  header.ptrOrData = ptrField->GetValueAsUnsigned(0, &ok);
  if (!ok)
    return false;
  header.lenOrData = lenField->GetValueAsUnsigned(0, &ok);
  if (!ok)
    return false;
  header.capacity = capField->GetValueAsUnsigned(0, &ok);
  return ok;
}

} // namespace

/// Render the active arm's payload into `out` — the stdlib `String`'s
/// lowered shape via `dumpMojoString`, or the child's registered summary
/// / raw scalar value for everything else.
///
/// TODO(MOCO-3787): drop the `String` branch (and the `MojoStringHelpers`
/// shared helper) once DWARF debug-info emission preserves the
/// pre-lowering types inside `!pop.union` members — the registered String /
/// List / user-type summary providers will then fire on the active arm via
/// the generic `GetSummaryAsCString` path below with no per-type code here.
static void renderActivePayload(ValueObject &activeValue,
                                llvm::StringRef activeFQN,
                                const TypeSummaryOptions &options,
                                StreamString &out) {
  // Match on the fully-qualified name to avoid colliding with user-defined
  // types also named `String` in a different namespace.
  if (activeFQN == kStringFQN) {
    MojoStringHeader header{};
    if (!readLoweredStringHeader(activeValue.GetSP(), header))
      return;
    // If `dumpMojoString` fails part-way through (e.g. the target read
    // can't be satisfied), discard whatever partial bytes it wrote so the
    // caller falls back to the type-name-only rendering instead of
    // wrapping garbage in `String(…)`.
    if (!dumpMojoString(activeValue, header, out, options))
      out.Clear();
    return;
  }

  std::string childSummary;
  activeValue.GetSummaryAsCString(childSummary, options);
  if (!childSummary.empty()) {
    out << childSummary;
    return;
  }
  if (const char *raw = activeValue.GetValueAsCString())
    out << raw;
}

/// Render `activeValue` into `stream` as `(payload)`, writing nothing if the
/// payload cannot be decoded. Shared by the Optional and Variant providers.
static void appendPayloadIfPresent(ValueObject &activeValue,
                                   llvm::StringRef activeFQN,
                                   const TypeSummaryOptions &options,
                                   Stream &stream) {
  StreamString payload;
  renderActivePayload(activeValue, activeFQN, options, payload);
  if (!payload.Empty())
    stream.Format("({0})", payload.GetString());
}

bool M::KGEN::Mojo::mojoOptionalSummaryProvider(ValueObject &valobj,
                                                Stream &stream,
                                                TypeSummaryOptions options) {
  // Optional[T] stores `_value: Variant[_NoneType, T]`.
  // Discriminant 0 = _NoneType (None), discriminant 1 = T (Some value).
  //
  // Note: Optional[T] where T is UnsafeNicheable uses _NichedOptionalStorage,
  // which has no `_impl` field. parseVariantInfo returns {} in that case, so
  // this formatter falls back to LLDB's default display for those types (e.g.
  // Optional[UnsafePointer[T]]). See _NichedOptionalStorage in variant.mojo.
  ValueObjectSP obj = valobj.GetNonSyntheticValue();
  if (!obj)
    return false;

  ValueObjectSP innerVariant = nonSyntheticChild(*obj, "_value");
  if (!innerVariant)
    return false;

  auto info = parseVariantInfo(innerVariant);
  if (!info)
    return false;

  if (info->discriminant == 0) {
    stream << "None";
    return true;
  }

  // Defensive: Optional is a 2-arm Variant; discriminant 0 is handled above,
  // so discriminant must be 1 here. Guard against misapplication to other
  // types.
  if (info->discriminant != 1)
    return false;

  // Emit `Some` without `(...)` when there is nothing safe to put inside:
  // either the union child was not exposed to LLDB (activeValue is null) or
  // the payload decoded to an empty string.
  stream << "Some";

  if (info->activeValue)
    appendPayloadIfPresent(*info->activeValue, info->typeNames[1].fullName,
                           options, stream);
  return true;
}

bool M::KGEN::Mojo::mojoVariantSummaryProvider(ValueObject &valobj,
                                               Stream &stream,
                                               TypeSummaryOptions options) {
  auto info = parseVariantInfo(valobj.GetSP());
  if (!info)
    return false;

  const VariantTypeName &active = info->typeNames[info->discriminant];
  stream << active.displayName;

  if (!info->activeValue)
    return true;

  appendPayloadIfPresent(*info->activeValue, active.fullName, options, stream);
  return true;
}
