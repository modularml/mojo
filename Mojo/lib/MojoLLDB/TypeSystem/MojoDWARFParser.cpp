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

#include "MojoDWARFParser.h"
#include "Mojo/KGENDialect/DebugInfoEncoding.h"
#include "Mojo/KGENDialect/KGENDType.h"
#include "Mojo/MojoTooling/PublicASTDecl.h"
#include "MojoTypeSystem.h"
#include "Support/ErrorOr.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/source/Plugins/SymbolFile/DWARF/DWARFDIE.h"
#include "lldb/source/Plugins/SymbolFile/DWARF/DWARFUnit.h"
#include "lldb/source/Plugins/SymbolFile/DWARF/LogChannelDWARF.h"
#include <filesystem>

using namespace M;
using namespace M::KGEN::Mojo;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::plugin::dwarf;
using namespace llvm::dwarf;

/// Implementation of `generateFunctionDisplayName` that recursively traverses
/// the scopes in the source name, generating a human readable version of the
/// function.
static void doGenerateFunctionDisplayName(DebugInfo::SourceNameAttr attr,
                                          ArrayRef<StringAttr> &paramValues,
                                          llvm::raw_ostream &os) {
  if (DebugInfo::SourceNameAttr parent = attr.getParent()) {
    doGenerateFunctionDisplayName(parent, paramValues, os);
    os << "::";
  }

  os << attr.getName().getValue();
  if (!attr.getParamTypes().empty()) {
    os << "<";
    ArrayRef<DebugInfo::SourceNameAttr> paramTypes = attr.getParamTypes();
    size_t paramDisplayCount = std::min(paramTypes.size(), paramValues.size());
    llvm::interleaveComma(paramTypes.take_front(paramDisplayCount), os,
                          [&](DebugInfo::SourceNameAttr paramType) {
                            StringRef paramValue =
                                paramValues.front().getValue();
                            paramValue.consume_front(":type ");
                            os << paramValue;
                            paramValues = paramValues.drop_front(1);
                          });
    // FIXME(25047): if we don't have all required parameters available, we
    // just show `...`
    if (paramDisplayCount < paramTypes.size()) {
      if (paramDisplayCount > 0)
        os << ", ";
      os << "...";
    }
    os << ">";
  }
}

/// Generate a human readable version of a function given its SourceName.
static std::string generateFunctionDisplayName(DebugInfo::SourceNameAttr attr) {
  ArrayRef<StringAttr> paramValues = attr.getParamValues();
  // FIXME(25047): If we are in a nested function, we don't show parameters.
  if (attr.getParent() &&
      attr.getParent().getKind() == DebugInfo::SourceNameKind::Fn)
    paramValues = {};

  std::string displayName;
  llvm::raw_string_ostream os(displayName);
  doGenerateFunctionDisplayName(attr, paramValues, os);

  if (attr.getKind() == DebugInfo::SourceNameKind::Fn) {
    // TODO(25048): we need to figure out a nice way to include the arguments
    // of functions. For now we just show `...` for the leaf function. This is
    // fine for dumping stack traces because LLDB will replace the `...` of the
    // leaf function with the actual arguments from DWARF as part of the
    // `SBFrame.GetDescription()` invocation.
    // However, we can't show the arguments of parent functions, because then
    // LLDB would place the DWARF variables there instead of in the leaf
    // function.
    os << "(";
    if (!attr.getArgTypes().empty())
      os << "...";
    os << ")";
  }
  return displayName;
}

/// Bag of data for all the attributes parsed from a DWARF entry.
struct ParsedDWARFTypeAttributes {
  explicit ParsedDWARFTypeAttributes(const DWARFDIE &die) {
    DWARFAttributes attributes = die.GetAttributes();
    for (size_t i = 0; i < attributes.Size(); ++i) {
      dw_attr_t attr = attributes.AttributeAtIndex(i);
      DWARFFormValue formValue;
      if (!attributes.ExtractFormValueAtIndex(i, formValue))
        continue;
      switch (attr) {
      default:
        break;

      case DW_AT_byte_size:
        byteSize = formValue.Unsigned();
        break;

      case DW_AT_alignment:
        alignment = formValue.Unsigned();
        break;

      case DW_AT_decl_file:
        decl.SetFile(
            attributes.CompileUnitAtIndex(i)->GetFile(formValue.Unsigned()));
        break;

      case DW_AT_decl_line:
        decl.SetLine(formValue.Unsigned());
        break;

      case DW_AT_decl_column:
        decl.SetColumn(formValue.Unsigned());
        break;

      case DW_AT_encoding:
        encoding = formValue.Unsigned();
        break;

      case DW_AT_external:
        external = formValue.Boolean();
        break;

      case DW_AT_inline:
        inlined = true;
        break;

      case DW_AT_linkage_name:
        linkageName.SetCString(formValue.AsCString());
        break;

      case DW_AT_name:
        name.SetCString(formValue.AsCString());
        break;

      case DW_AT_type:
        type = formValue;
        break;

      case DW_AT_GNU_vector:
        isVector = true;
        break;
      }
    }
  }

  bool inlined = false;
  lldb_private::ConstString linkageName;
  bool external = false;
  DWARFFormValue type;
  lldb_private::ConstString name;
  lldb_private::Declaration decl;
  std::optional<uint64_t> byteSize;
  std::optional<uint64_t> alignment;
  uint32_t encoding = 0;
  bool isVector = false;
};

/// Bag of data for all the attributes parsed from a struct member DWARF entry.
struct MemberAttributes {
  MemberAttributes(const DWARFDIE &die, const DWARFDIE &parentDie) {
    DWARFAttributes attributes = die.GetAttributes();
    for (size_t i = 0; i < attributes.Size(); ++i) {
      const dw_attr_t attr = attributes.AttributeAtIndex(i);
      DWARFFormValue formValue;
      if (attributes.ExtractFormValueAtIndex(i, formValue)) {
        switch (attr) {
        case DW_AT_name:
          name = formValue.AsCString();
          break;
        case DW_AT_type:
          type = formValue;
          break;
        case DW_AT_data_member_location:
          byteOffset = formValue.Unsigned();
          break;
        default:
          break;
        }
      }
    }
  }
  const char *name = nullptr;
  std::optional<uint64_t> byteOffset;
  DWARFFormValue type;
};

MojoDWARFParser::MojoDWARFParser(MojoTypeSystem &typeSystem)
    : DWARFASTParser(Kind::DWARFASTParserClang), typeSystem(typeSystem) {}

MojoDWARFParser::~MojoDWARFParser() = default;

MojoASTDeclRef MojoDWARFParser::getDeclForDIE(const DWARFDIE &die) {
  if (!die)
    return {};
  if (MojoASTDeclRef decl = getCachedDeclForDIE(die))
    return decl;

  SymbolFileDWARF *dwarf = die.GetDWARF();
  MojoASTDeclRef decl;

  switch (die.Tag()) {
  case DW_TAG_compile_unit: {
    ParsedDWARFTypeAttributes attrs(die);
    std::string name;
    // The DIE name is the file path
    if (!attrs.name.IsEmpty()) {
      // The name of the module is the base name of the file. We might need to
      // eventually define a unique parser for each compile unit in case of name
      // collision.
      name = std::filesystem::path(attrs.name.AsCString(/*value_if_empty=*/""))
                 .stem()
                 .string();
    }

    decl = typeSystem.getOrCreateDeclChainForDie(die, name);
    break;
  }
  case DW_TAG_inlined_subroutine:
  case DW_TAG_subroutine_type:
  case DW_TAG_subprogram: {
    ParsedDWARFTypeAttributes attrs(die);
    decl = typeSystem.getOrCreateDeclChainForDie(die, attrs.name);
    break;
  }
  case DW_TAG_structure_type: {
    ParsedDWARFTypeAttributes attrs(die);
    decl = typeSystem.getOrCreateDeclChainForDie(die, attrs.name);
    break;
  }
  default:
    dwarf->GetObjectFile()->GetModule()->ReportError(
        "[MojoDWARFParser::getDeclForDIE]: Unhandled type tag. Die = {0:x}, "
        "tag = {1}.",
        die.GetOffset(), DW_TAG_value_to_name(die.Tag()));
    break;
  }

  if (decl)
    dieToDecl[die.GetDIE()] = decl;

  return decl;
}

MojoASTDeclRef MojoDWARFParser::getCachedDeclForDIE(const DWARFDIE &die) {
  if (die) {
    if (auto pos = dieToDecl.find(die.GetDIE()); pos != dieToDecl.end())
      return pos->second;
  }
  return {};
}

MojoASTDeclRef
MojoDWARFParser::getDeclContextContainingDIE(const DWARFDIE &die,
                                             DWARFDIE *declDieCopy) {
  SymbolFileDWARF *dwarf = die.GetDWARF();

  DWARFDIE declDie = dwarf->GetDeclContextDIEContainingDIE(die);

  if (declDieCopy)
    *declDieCopy = declDie;

  if (declDie) {
    if (MojoASTDeclRef decl = getDeclForDIE(declDie))
      return decl;
  }
  return {};
}

void MojoDWARFParser::updateSymbolContextScopeForType(const SymbolContext &sc,
                                                      const DWARFDIE &die,
                                                      TypeSP &type) {
  assert(type->GetFullCompilerType().IsValid() &&
         "All types created from DWARF must be valid.");

  DWARFDIE scParentDie = SymbolFileDWARF::GetParentSymbolContextDIE(die);
  dw_tag_t scParentTag = scParentDie.Tag();

  SymbolContextScope *symbolContextScope = nullptr;
  if (scParentTag == DW_TAG_compile_unit) {
    symbolContextScope = sc.comp_unit;
  } else if (sc.function != nullptr && scParentDie) {
    symbolContextScope =
        sc.function->GetBlock(true).FindBlockByID(scParentDie.GetID());
    if (symbolContextScope == nullptr)
      symbolContextScope = sc.function;
  } else {
    symbolContextScope = sc.module_sp.get();
  }

  if (symbolContextScope != nullptr)
    type->SetSymbolContextScope(symbolContextScope);
}

CompilerDeclContext
MojoDWARFParser::GetDeclContextForUIDFromDWARF(const DWARFDIE &die) {
  if (die.Tag() == DW_TAG_compile_unit) {
    if (MojoASTDeclRef decl = getDeclForDIE(die))
      return CompilerDeclContext(&typeSystem, &decl);
  }
  return {};
}

TypeSP
MojoDWARFParser::ParseTypeFromDWARF(const lldb_private::SymbolContext &sc,
                                    const DWARFDIE &die, bool *typeIsNewPtr) {
  if (typeIsNewPtr)
    *typeIsNewPtr = false;

  if (!die)
    return {};

  SymbolFileDWARF *dwarf = die.GetDWARF();
  lldb_private::Type *typePtr = dwarf->GetDIEToType().lookup(die.GetDIE());
  if (typePtr == DIE_IS_BEING_PARSED)
    return nullptr;
  if (typePtr)
    return typePtr->shared_from_this();
  // Set a bit that lets us know that we are currently parsing this.
  dwarf->GetDIEToType()[die.GetDIE()] = DIE_IS_BEING_PARSED;

  ParsedDWARFTypeAttributes attrs(die);

  if (Log *log = GetLog(DWARFLog::TypeCompletion | DWARFLog::Lookups)) {
    dwarf->GetObjectFile()->GetModule()->LogMessage(
        log,
        "[MojoDWARFParser::ParseTypeFromDWARF] Will parse type. Die = {0:x}, "
        "tag = {1}, name = '{2}', linkage name = '{3}', byte size = {4}.",
        die.GetOffset(), DW_TAG_value_to_name(die.Tag()), die.GetName(),
        attrs.linkageName.AsCString(/*value_if_empty=*/""), attrs.byteSize);
  }

  if (typeIsNewPtr)
    *typeIsNewPtr = true;

  const dw_tag_t tag = die.Tag();
  TypeSP type;

  switch (tag) {
  case DW_TAG_pointer_type: {
    // Pointer types are lazily resolved by LLDB, which handles cases of pointee
    // types that failed parsing.
    type = dwarf->MakeType(die.GetID(), attrs.name, attrs.byteSize, nullptr,
                           attrs.type.Reference().GetID(),
                           lldb_private::Type::eEncodingIsPointerUID,
                           &attrs.decl, CompilerType(),
                           lldb_private::Type::ResolveState::Unresolved);
    break;
  }
  case DW_TAG_unspecified_type: {
    // For clang, these are often `nullptr_t` or `decltype(nullptr)`. However,
    // if a different name is found, clang attempts to parse this unspecified as
    // a base type using its encoding, which might fail anyway. We follow that
    // approach.
    CompilerType mojoType;
    if (attrs.name == "kgen.dtype.invalid")
      mojoType = typeSystem.createCompilerTypeFromDType(attrs.name);
    else if ((attrs.name == "void" || attrs.name == "opaque"))
      mojoType = typeSystem.getBuiltinTypeFromMLIRTypeName("!kgen.none");

    if (mojoType.IsValid()) {
      type =
          dwarf->MakeType(die.GetID(), attrs.name,
                          attrs.byteSize.value_or(
                              llvm::expectedToOptional(
                                  mojoType.GetByteSize(/*exe_scope=*/nullptr))
                                  .value_or(0)),
                          nullptr, attrs.type.Reference().GetID(),
                          lldb_private::Type::eEncodingIsUID, &attrs.decl,
                          mojoType, lldb_private::Type::ResolveState::Full);
      break;
    }
    dwarf->GetObjectFile()->GetModule()->ReportError(
        "[MojoDWARFParser::ParseTypeFromDWARF] Unexpected unspecified type. "
        "Die = {0:x}, tag = {1}, name = '{2}'.",
        die.GetOffset(), DW_TAG_value_to_name(die.Tag()), die.GetName());
    [[fallthrough]];
  }

  case DW_TAG_base_type: {
    if (attrs.byteSize.value_or(0) == 0) {
      dwarf->GetObjectFile()->GetModule()->ReportError(
          "[MojoDWARFParser::ParseTypeFromDWARF] Builtin type with 0 byte "
          "size. Die = {0:x}, tag = {1}, name = '{2}'.",
          die.GetOffset(), DW_TAG_value_to_name(die.Tag()), die.GetName());
      break;
    }
    CompilerType mojoType = typeSystem.getBuiltinScalarType(
        attrs.name.GetStringRef(), attrs.encoding, attrs.byteSize.value_or(0));
    if (!mojoType.IsValid()) {
      dwarf->GetObjectFile()->GetModule()->ReportError(
          "[MojoDWARFParser::ParseTypeFromDWARF] Couldn't create builtin type. "
          "Die = {0:x}, tag = {1}, name = '{2}'.",
          die.GetOffset(), DW_TAG_value_to_name(die.Tag()), die.GetName());
    }
    type = dwarf->MakeType(die.GetID(), attrs.name, attrs.byteSize, nullptr,
                           attrs.type.Reference().GetID(),
                           lldb_private::Type::eEncodingIsUID, &attrs.decl,
                           mojoType, lldb_private::Type::ResolveState::Full);

    break;
  }
  case DW_TAG_inlined_subroutine:
  case DW_TAG_subroutine_type:
  case DW_TAG_subprogram: {
    if (MojoASTDeclRef decl = getDeclForDIE(die)) {
      CompilerType mojoType = typeSystem.createCompilerType(decl.getType());

      type = dwarf->MakeType(die.GetID(), attrs.name, attrs.byteSize, nullptr,
                             attrs.type.Reference().GetID(),
                             lldb_private::Type::eEncodingIsUID, &attrs.decl,
                             mojoType, lldb_private::Type::ResolveState::Full);
    }
    break;
  }
  case DW_TAG_array_type: {
    DWARFDIE elementTypeDie = attrs.type.Reference();
    ParsedDWARFTypeAttributes elementTypeAttrs(elementTypeDie);
    lldb_private::Type *elementType =
        die.ResolveTypeUID(attrs.type.Reference());
    std::optional<SymbolFile::ArrayInfo> arrayInfo = ParseChildArrayInfo(die);

    if (elementType && arrayInfo && !arrayInfo->element_orders.empty()) {
      assert(arrayInfo->element_orders.front().has_value());
      size_t numElements = *arrayInfo->element_orders.front();

      CompilerType mojoType =
          attrs.isVector
              ? typeSystem.createSIMDType(elementTypeDie.GetName(), numElements)
              : typeSystem.createPOPArrayType(
                    elementType->GetFullCompilerType().GetOpaqueQualType(),
                    numElements);

      if (mojoType.IsValid()) {
        type = dwarf->MakeType(
            die.GetID(), ConstString(),
            attrs.byteSize.value_or(
                llvm::expectedToOptional(
                    mojoType.GetByteSize(/*exe_ctx=*/nullptr))
                    .value_or(0)),
            nullptr, die.GetID(), lldb_private::Type::eEncodingIsUID,
            &attrs.decl, mojoType, lldb_private::Type::ResolveState::Full);
        type->SetEncodingType(elementType);
      }
    }
    if (!type) {
      dwarf->GetObjectFile()->GetModule()->ReportError(
          "The array type '{0}' at offset {1:x} with element type '{2}' "
          "couldn't be parsed.",
          attrs.name.AsCString(/*value_if_empty=*/""), die.GetOffset(),
          elementTypeDie.GetName());
    }

    break;
  }
  case DW_TAG_structure_type: {
    // Several builtin types like !kgen.string are encoded as structs. We can
    // just parse them as regular MLIR types instead of traversing their DWARF.
    // At least in the specific case of primitive types like !kgen.string, it
    // will allow us to format them correctly because the corresponding printers
    // are type-based and not name-based.
    CompilerType mojoType =
        typeSystem.getBuiltinTypeFromMLIRTypeName(attrs.name);
    // If we recover the type, we need to make sure that the encoded byte size
    // matches the one from MLIR. If there's a mismatch, then either the debug
    // info is wrong or the MLIR type in this version of the parser is different
    // from the one that produced the debug info, in which case we discard the
    // MLIR type and do regular DWARF parsing.
    if (mojoType.IsValid()) {
      llvm::Expected<uint64_t> mlirByteSize =
          mojoType.GetByteSize(/*exe_scope=*/nullptr);
      if (!mlirByteSize) {
        mojoType = {};
        dwarf->GetObjectFile()->GetModule()->ReportError(
            "The parsed MLIR structure type '{0}' has not byte size. The "
            "MLIR type won't be used and regular MLIR-agnostic DWARF parsing "
            "will be performed. Error: {1}",
            attrs.name.AsCString(/*value_if_empty=*/nullptr),
            llvm::toString(mlirByteSize.takeError()));
      } else if (attrs.byteSize && *attrs.byteSize != *mlirByteSize) {
        mojoType = {};
        dwarf->GetObjectFile()->GetModule()->ReportError(
            "The parsed MLIR structure type '{0}' has a different size ({1}) "
            "than the one in the debug info ({2}). The MLIR type won't be used "
            "and regular MLIR-agnostic DWARF parsing will be performed.",
            attrs.name.AsCString(/*value_if_empty=*/nullptr), *mlirByteSize,
            *attrs.byteSize);
      } else {
        type = dwarf->MakeType(
            die.GetID(), attrs.name, attrs.byteSize, nullptr, LLDB_INVALID_UID,
            lldb_private::Type::eEncodingIsUID, &attrs.decl, mojoType,
            lldb_private::Type::ResolveState::Full);
      }
    }
    if (!mojoType.IsValid()) {
      if (MojoASTDeclRef decl = getDeclForDIE(die)) {
        CompilerType mojoType = typeSystem.createCompilerType(decl.getType());
        type = dwarf->MakeType(
            die.GetID(), attrs.name, attrs.byteSize, nullptr, LLDB_INVALID_UID,
            lldb_private::Type::eEncodingIsUID, &attrs.decl, mojoType,
            lldb_private::Type::ResolveState::Full);
        // FIXME(23821): We need to complete the struct right away here because
        // the generic dwarf parser uses the clang typesystem to complete types,
        // which obviously wouldn't work for us. We'll eventually fix this,
        // which will make the dwarf parser lazy.
        CompleteTypeFromDWARF(die, type.get(), mojoType);
      }
    }
    break;
  }
  default:
    dwarf->GetObjectFile()->GetModule()->ReportError(
        "[MojoDWARFParser::ParseTypeFromDWARF]: Unhandled type tag. "
        "Die = {0:x}, tag = {1}, name = {2}",
        die.GetOffset(), tag, DW_TAG_value_to_name(die.Tag()), die.GetName());
    break;
  }

  if (!type) {
    type = dwarf->MakeType(
        die.GetID(), attrs.name, attrs.byteSize, nullptr,
        attrs.type.Reference().GetID(), lldb_private::Type::eEncodingIsUID,
        &attrs.decl, CompilerType(), lldb_private::Type::ResolveState::Full);
  }

  updateSymbolContextScopeForType(sc, die, type);
  dwarf->GetDIEToType()[die.GetDIE()] = type.get();
  return type;
}

bool MojoDWARFParser::CompleteStructureTypeFromDWARF(
    const DWARFDIE &die, lldb_private::Type *type,
    const CompilerType &compilerType) {
  MojoASTDeclRef structDecl = getDeclForDIE(die);
  assert(structDecl && "All structs should have a decl.");

  if (completedDecls.contains(&*structDecl))
    return true;

  SymbolFileDWARF *dwarf = die.GetDWARF();

  for (DWARFDIE memberDie : die.children()) {
    if (memberDie.Tag() == DW_TAG_member) {
      MemberAttributes attrs(memberDie, die);
      lldb_private::Type *memberType =
          die.ResolveTypeUID(attrs.type.Reference());
      if (!memberType) {
        dwarf->GetObjectFile()->GetModule()->ReportError(
            "[MojoDWARFParser::CompleteTypeFromDWARF]: Couldn't complete "
            "the struct type '{0}' because one of its fields couldn't be "
            "parsed. Die = {1:x}, memberDie = {2:x}.",
            die.GetName(), die.GetOffset(), memberDie.GetOffset());
        return false;
      }

      llvm::Expected<uint64_t> typeSize = memberType->GetByteSize(nullptr);
      if (!typeSize) {
        dwarf->GetObjectFile()->GetModule()->ReportError(
            "[MojoDWARFParser::CompleteTypeFromDWARF]: Couldn't complete "
            "the struct type '{0}' because one of its fields has no size. Die "
            "= {1:x}, member die = {2:x}: {3}",
            die.GetName(), die.GetOffset(), memberDie.GetOffset(),
            llvm::toString(typeSize.takeError()));
        return false;
      }
      typeSystem.addFieldToStruct(
          structDecl, attrs.name,
          memberType->GetFullCompilerType().GetOpaqueQualType());
    }
  }
  ParsedDWARFTypeAttributes attrs(die);
  if (attrs.byteSize) {
    llvm::Expected<uint64_t> mlirByteSize = compilerType.GetByteSize(
        /*exe_scope=*/nullptr);
    if (!mlirByteSize) {
      dwarf->GetObjectFile()->GetModule()->ReportError(
          "[MojoDWARFParser::CompleteTypeFromDWARF]: {0}",
          llvm::toString(mlirByteSize.takeError()));
      return false;
    }
    if (mlirByteSize.get() != attrs.byteSize) {
      dwarf->GetObjectFile()->GetModule()->ReportError(
          "[MojoDWARFParser::CompleteTypeFromDWARF]: The struct type '{0}' "
          "doesn't have the same size as reported in the DWARF after type "
          "completion. Die = {1:x}.",
          die.GetName(), die.GetOffset());
      return false;
    }
  }
  completedDecls.insert(&*structDecl);
  return true;
}

bool MojoDWARFParser::CompleteTypeFromDWARF(const DWARFDIE &die,
                                            lldb_private::Type *type,
                                            const CompilerType &compilerType) {
  if (!die)
    return false;

  if (die.Tag() == DW_TAG_structure_type)
    return CompleteStructureTypeFromDWARF(die, type, compilerType);

  SymbolFileDWARF *dwarf = die.GetDWARF();
  dwarf->GetObjectFile()->GetModule()->ReportError(
      "[MojoDWARFParser::CompleteTypeFromDWARF]: Couldn't complete die. Die = "
      "{0:x}, tag = {1}.",
      die.GetOffset(), DW_TAG_value_to_name(die.Tag()));
  return false;
}

DebugInfo::SourceNameAttr
MojoDWARFParser::extractSourceName(const DWARFDIE &die) {
  for (auto child : die.children()) {
    if (child.Tag() == DW_TAG_LLVM_annotation) {
      StringRef tagName, tagValue;
      DWARFAttributes attributes = child.GetAttributes();
      for (size_t i = 0, e = attributes.Size(); i < e; ++i) {
        DWARFFormValue formValue;
        const dw_attr_t attr = attributes.AttributeAtIndex(i);
        if (attributes.ExtractFormValueAtIndex(i, formValue)) {
          switch (attr) {
          case DW_AT_name:
            tagName = formValue.AsCString();
            break;
          case DW_AT_const_value:
            tagValue = formValue.AsCString();
            break;
          default:
            break;
          }
        }
      }
      if (tagName == "mojo_source_name") {
        ErrorOr<DebugInfo::SourceNameAttr> attrOr =
            DebugInfo::SourceNameAttr::decode(typeSystem.getMLIRContext(),
                                              tagValue);
        if (succeeded(attrOr))
          return *attrOr;
      }
    }
  }
  return DebugInfo::SourceNameAttr();
}

Function *MojoDWARFParser::ParseFunctionFromDWARF(CompileUnit &comp_unit,
                                                  const DWARFDIE &die,
                                                  AddressRanges funcRange) {
  Log *log = GetLog(DWARFLog::TypeCompletion | DWARFLog::Lookups);
  SymbolFileDWARF *dwarf = die.GetDWARF();

  if (log) {
    dwarf->GetObjectFile()->GetModule()->LogMessage(
        log,
        "[MojoDWARFParser::ParseFunctionFromDWARF] Will parse function. Die = "
        "{0:x}, tag = {1}, name = '{2}'.",
        die.GetOffset(), DW_TAG_value_to_name(die.Tag()), die.GetName());
  }

  auto doWork = [&]() -> Function * {
    assert(funcRange.front().GetBaseAddress().IsValid());

    llvm::DWARFAddressRangesVector funcRanges;
    const char *name = nullptr;
    const char *mangled = nullptr;
    std::optional<int> declFile;
    std::optional<int> declLine;
    std::optional<int> declColumn;
    std::optional<int> callFile;
    std::optional<int> callLine;
    std::optional<int> callColumn;
    DWARFExpressionList frameBase;

    const dw_tag_t tag = die.Tag();

    if (tag != DW_TAG_subprogram)
      return nullptr;

    if (die.GetDIENamesAndRanges(name, mangled, funcRanges, declFile, declLine,
                                 declColumn, callFile, callLine, callColumn,
                                 &frameBase)) {
      Mangled funcName;
      if (mangled)
        funcName.SetMangledName(ConstString(mangled));
      else
        funcName.SetMangledName(ConstString(name));

      // If the name is a SourceName, then generate a human readable version of
      // it, otherwise we keep the name unchanged as its display version.
      if (DebugInfo::SourceNameAttr sourceName = extractSourceName(die)) {
        funcName.SetDemangledName(
            ConstString(generateFunctionDisplayName(sourceName)));
      } else {
        funcName.SetDemangledName(ConstString(name));
      }

      FunctionSP func;
      std::unique_ptr<Declaration> decl;
      if (declFile || declLine || declColumn)
        decl = std::make_unique<Declaration>(
            die.GetCU()->GetFile(declFile ? *declFile : 0),
            declLine ? *declLine : 0, declColumn ? *declColumn : 0);

      SymbolFileDWARF *dwarf = die.GetDWARF();
      // Supply the type only if it has already been parsed
      lldb_private::Type *funcType = dwarf->GetDIEToType().lookup(die.GetDIE());

      assert(funcType == nullptr || funcType != DIE_IS_BEING_PARSED);

      const user_id_t funcUID = die.GetID();
      func = std::make_shared<Function>(&comp_unit,
                                        funcUID, // UserID is the DIE offset
                                        funcUID, funcName, funcType,
                                        funcRange.front().GetBaseAddress(),
                                        funcRange); // first address range

      if (func.get() != nullptr) {
        if (frameBase.IsValid())
          func->GetFrameBaseExpression() = frameBase;
        comp_unit.AddFunction(func);
        return func.get();
      }
    }
    return nullptr;
  };
  auto func = doWork();
  if (!func) {
    dwarf->GetObjectFile()->GetModule()->ReportError(
        "[MojoDWARFParser::ParseFunctionFromDWARF] failed to create a "
        "function. Die = {0:x}, tag = {1}, name = '{2}'.",
        die.GetOffset(), DW_TAG_value_to_name(die.Tag()), die.GetName());
  }
  return func;
}
