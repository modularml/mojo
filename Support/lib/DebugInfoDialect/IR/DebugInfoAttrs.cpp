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

#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/DebugInfoDialect/IR/DebugInfoInterfaces.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/DebugInfoDialect/IR/DebugInfoTypes.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
#include <functional>
#include <iterator>
#include <optional>
#include <string>
#include <utility>

using namespace M;
using namespace M::DebugInfo;

//===----------------------------------------------------------------------===//
// DebugInfoDialect
//===----------------------------------------------------------------------===//

void DebugInfoDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// SourceNameAttr
//===----------------------------------------------------------------------===//

StringAttr SourceNameAttr::encode() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  encode(os);
  return StringAttr::get(getContext(), str);
}

void SourceNameAttr::encode(llvm::raw_ostream &os) const {
  // Aim to create a name like `builtin::int::Int`.
  if (getParent()) {
    getParent().encode(os);
    os << "::";
  }

  // The only character not allowed in Mojo symbol names is backtick.
  auto printString = [&](StringRef str) {
    if (llvm::all_of(str, [](char c) { return std::isalnum(c) || c == '_'; }))
      os << str;
    else
      os << '`' << str << '`';
  };

  // Base name.
  if (getKind() != SourceNameKind::Unknown)
    os << stringifySourceNameKind(getKind()) << ' ';

  printString(getName());

  // Parameter types.
  if (!getParamTypes().empty()) {
    os << '[';
    llvm::interleave(
        getParamTypes(), os, [&](SourceNameAttr type) { type.encode(os); },
        ",");
    os << ']';
  }

  // Argument types.
  if (!getArgTypes().empty()) {
    os << '(';
    llvm::interleave(
        getArgTypes(), os, [&](SourceNameAttr type) { type.encode(os); }, ",");
    os << ')';
  }

  // Parameter values.
  if (!getParamValues().empty()) {
    os << '<';
    llvm::interleave(
        getParamValues(), os, [&](StringAttr value) { printString(value); },
        ",");
    os << '>';
  }

  // Decorator values.
  if (!getDecorators().empty()) {
    os << " @[";
    llvm::interleave(
        getDecorators(), os, [&](SourceNameAttr dec) { dec.encode(os); }, ",");
    os << "]";
  }
}

namespace {
/// Source name parser, implementing the opposite of the encoding logic above.
class SourceNameParser {
public:
  SourceNameParser(MLIRContext *ctx, StringRef buf)
      : ctx(ctx), cur(buf.begin()), end(buf.end()) {}

  /// Parse a string as a sequence of alnum characters or between backticks.
  ErrorOr<StringRef> parseString();
  /// Parse a whole source name.
  ErrorOr<SourceNameAttr> parseSourceName();
  /// Optionally parse a character if it's the next one.
  bool parseOptional(char c) {
    if (*cur != c)
      return false;
    ++cur;
    return true;
  }
  /// Parse and consume a character.
  ErrorOrSuccess parse(char c) {
    if (*cur != c)
      return Error("expected a '" + StringRef(&c, 1) + "'");
    ++cur;
    return success();
  }
  /// Parse an optional comma-separated list with a delimiter.
  ErrorOrSuccess parseList(char open, char close,
                           function_ref<ErrorOrSuccess()> parseFn) {
    if (!parseOptional(open))
      return success();
    do {
      if (auto err = parseFn())
        return err.takeError();
    } while (parseOptional(','));
    return parse(close);
  }
  ErrorOrSuccess parseList(StringRef open, StringRef close,
                           function_ref<ErrorOrSuccess()> parseFn) {
    bool commit = false;
    for (char c : open) {
      if (!commit) {
        if (!parseOptional(c))
          return success();
        commit = true;
      } else {
        if (auto err = parse(c))
          return err.takeError();
      }
    }
    do {
      if (auto err = parseFn())
        return err.takeError();
    } while (parseOptional(','));
    for (char c : close) {
      if (auto err = parse(c))
        return err.takeError();
    }
    return success();
  }

private:
  ErrorOrSuccess
  parseSourceNameImpl(StringAttr &baseName, SourceNameKind &kind,
                      SmallVectorImpl<SourceNameAttr> &paramTypes,
                      SmallVectorImpl<SourceNameAttr> &argTypes,
                      SmallVectorImpl<StringAttr> &paramValues,
                      SmallVectorImpl<SourceNameAttr> &decoratorValues);

  MLIRContext *ctx;
  /// The current offset into the buffer.
  const char *cur;
  /// The buffer end marker.
  const char *end;
};
} // namespace

ErrorOr<StringRef> SourceNameParser::parseString() {
  // If this is an escaped string, parse until the next backtick.
  if (*cur == '`') {
    const char *start = ++cur;
    while (cur != end && *cur != '`')
      ++cur;
    // Error if we hit EOF.
    if (cur == end)
      return Error("unterminated `-escaped string");
    return StringRef(start, std::distance(start, cur++));
  }
  // Parse the sequence of alnum characters.
  const char *start = cur;
  while (cur != end && (std::isalnum(*cur) || *cur == '_'))
    ++cur;
  return StringRef(start, std::distance(start, cur));
}

ErrorOrSuccess SourceNameParser::parseSourceNameImpl(
    StringAttr &baseName, SourceNameKind &kind,
    SmallVectorImpl<SourceNameAttr> &paramTypes,
    SmallVectorImpl<SourceNameAttr> &argTypes,
    SmallVectorImpl<StringAttr> &paramValues,
    SmallVectorImpl<SourceNameAttr> &decoratorValues) {
  // Base name.
  ErrorOr<StringRef> name = parseString();
  if (name.isError())
    return name.takeError();
  baseName = StringAttr::get(ctx, name.takeValue());

  if (parseOptional(' ')) {
    if (std::optional<SourceNameKind> kindOr =
            symbolizeSourceNameKind(baseName)) {
      kind = *kindOr;
    } else {
      return Error("Unexpected kind '" + baseName.str() + "'");
    }
    ErrorOr<StringRef> name = parseString();
    if (name.isError())
      return name.takeError();
    baseName = StringAttr::get(ctx, name.takeValue());
  }

  // Parameter types.
  auto parseParamType = [&]() -> ErrorOrSuccess {
    ErrorOr<SourceNameAttr> name = parseSourceName();
    if (name.isError())
      return name.takeError();
    paramTypes.push_back(name.takeValue());
    return success();
  };
  if (auto err = parseList('[', ']', parseParamType))
    return err.takeError();

  // Argument types.
  auto parseArgType = [&]() -> ErrorOrSuccess {
    ErrorOr<SourceNameAttr> name = parseSourceName();
    if (name.isError())
      return name.takeError();
    argTypes.push_back(name.takeValue());
    return success();
  };
  if (auto err = parseList('(', ')', parseArgType))
    return err.takeError();

  // Parameter values.
  auto parseParam = [&]() -> ErrorOrSuccess {
    ErrorOr<StringRef> value = parseString();
    if (value.isError())
      return value.takeError();
    paramValues.push_back(StringAttr::get(ctx, value.takeValue()));
    return success();
  };
  if (auto err = parseList('<', '>', parseParam))
    return err.takeError();

  // Decorator values.
  auto parseDecorator = [&]() -> ErrorOrSuccess {
    ErrorOr<SourceNameAttr> decorator = parseSourceName();
    if (decorator.isError())
      return decorator.takeError();
    decoratorValues.push_back(decorator.takeValue());
    return success();
  };
  if (auto err = parseList(" @[", "]", parseDecorator))
    return err.takeError();
  return success();
}

ErrorOr<SourceNameAttr> SourceNameParser::parseSourceName() {
  SourceNameAttr next;
  SmallVector<SourceNameAttr> paramTypes, argTypes;
  SmallVector<StringAttr> paramValues;
  SmallVector<SourceNameAttr> decoratorValues;
  do {
    // Re-use state to save memory allocations.
    paramTypes.clear();
    argTypes.clear();
    paramValues.clear();
    decoratorValues.clear();
    StringAttr baseName;
    SourceNameKind kind = {};
    if (auto err = parseSourceNameImpl(baseName, kind, paramTypes, argTypes,
                                       paramValues, decoratorValues))
      return err.takeError();

    // Generate the source name with the current name as the parent.
    next = SourceNameAttr::get(baseName, paramTypes, argTypes, paramValues,
                               next, kind, decoratorValues);
  } while (parseOptional(':') && parseOptional(':'));
  return next;
}

ErrorOr<SourceNameAttr> SourceNameAttr::decode(MLIRContext *ctx,
                                               StringRef str) {
  SourceNameParser parser(ctx, str);
  return parser.parseSourceName();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Support/DebugInfoDialect/IR/DebugInfoEnums.cpp.inc"
#include "Support/DebugInfoDialect/IR/DebugInfoExprAttrInterfaces.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.cpp.inc"

//===----------------------------------------------------------------------===//
// DIAttr
//===----------------------------------------------------------------------===//

bool DIAttr::classof(Attribute attr) {
  return llvm::isa<DebugInfoDialect>(attr.getDialect());
}

//===----------------------------------------------------------------------===//
// DIScopeAttr
//===----------------------------------------------------------------------===//

bool DIScopeAttr::classof(Attribute attr) {
  return llvm::isa<DICompileUnitAttr, DIFileAttr, DILocalScopeAttr>(attr);
}

//===----------------------------------------------------------------------===//
// DILocalScopeAttr
//===----------------------------------------------------------------------===//

bool DILocalScopeAttr::classof(Attribute attr) {
  return llvm::isa<DILexicalBlockAttr, DISubprogramAttr>(attr);
}

//===----------------------------------------------------------------------===//
// DIAggregatesIntoExprAttr
//===----------------------------------------------------------------------===//

LogicalResult
DIAggregatesIntoExprAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 DIExprAttr fieldExpr, unsigned index,
                                 Type type) {
  // Verify type consistency if the expression has been lowered into DITypes.
  if (auto structType = ::dyn_cast<DIStructType>(type)) {
    if (structType.getMembers().size() <= index)
      return emitError() << "field index out of bounds for struct type: "
                         << type;
    if (structType.getMembers()[index].getType() != fieldExpr.getType()) {
      return emitError()
             << "operand type does not match struct field type at index "
             << index;
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// DIDerefExprAttr
//===----------------------------------------------------------------------===//

DIExprAttr DIDerefExprAttr::get(DIExprAttr ptrExpr, Type type) {
  if (auto refExpr = ::dyn_cast<DIRefOfExprAttr>(ptrExpr))
    return refExpr.getValueExpr();
  return get(ptrExpr.getContext(), ptrExpr, type);
}

LogicalResult
DIDerefExprAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                        DIExprAttr ptrExpr, Type type) {
  // Verify type consistency if the expression has been lowered into DITypes.
  if (auto ptrType = ::dyn_cast<DIPointerType>(ptrExpr.getType())) {
    if (ptrType.getElementType() != type) {
      return emitError() << "result type is not the element type of the "
                            "operand DIPointerType";
    }
  } else if (auto ptrType = ::dyn_cast<DITargetIndependentPointerType>(
                 ptrExpr.getType())) {
    if (ptrType.getElementType() != type) {
      return emitError() << "result type is not the element type of the "
                            "operand DITargetIndependentPointerType";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// DIRefOfExprAttr
//===----------------------------------------------------------------------===//

DIExprAttr DIRefOfExprAttr::get(DIExprAttr valueExpr, Type type) {
  if (auto derefExpr = ::dyn_cast<DIDerefExprAttr>(valueExpr))
    return derefExpr.getPtrExpr();
  return DIRefOfExprAttr::get(valueExpr.getContext(), valueExpr, type);
}

//===----------------------------------------------------------------------===//
// DISubprogramAttr
//===----------------------------------------------------------------------===//

DISubprogramAttr DISubprogramAttr::cloneWith(SourceNameAttr sourceName,
                                             StringAttr linkageName,
                                             DISubroutineType type) const {
  return DebugInfo::DISubprogramAttr::get(
      getCompileUnit(), getScope(), sourceName, linkageName, getFile(),
      getLine(), getScopeLine(), getSubprogramFlags(), type ? type : getType());
}

//===----------------------------------------------------------------------===//
// DI Expression Support
//===----------------------------------------------------------------------===//
DebugInfo::DIExprLeafReplacer::DIExprLeafReplacer(
    std::function<ErrorOr<DIExprAttr>(Type)> conversionFunc)
    : leafReplacer(std::move(conversionFunc)) {
  replacer.addReplacement(
      [&](DIIRValueExprAttr irValue)
          -> std::optional<std::pair<Attribute, WalkResult>> {
        auto result = leafReplacer(irValue.getType());
        if (failed(result)) {
          currErrorMsg = result.getError();
          return std::make_pair(nullptr, WalkResult::skip());
        }

        auto conversionResult = result.get();
        if (conversionResult.getType() != irValue.getType()) {
          currErrorMsg = "Converter result type differs from input type.";
          return std::make_pair(nullptr, WalkResult::skip());
        }
        return std::make_pair(conversionResult, WalkResult::skip());
      });
}

ErrorOr<DIExprAttr> DebugInfo::DIExprLeafReplacer::apply(DIExprAttr expr) {
  currErrorMsg = {};
  auto newExpr = dyn_cast_or_null<DIExprAttr>(replacer.replace(expr));
  if (!currErrorMsg.empty())
    return Error(currErrorMsg);
  if (!newExpr)
    return Error("LeafReplacer failed to replace.");
  return newExpr;
}

//===----------------------------------------------------------------------===//
// DI Scope Support
//===----------------------------------------------------------------------===//

WalkResult DebugInfo::walkLocation(Location loc, LocWalkPolicy policy,
                                   function_ref<WalkResult(Location)> walkFn) {
  WalkResult locWalkResult = walkFn(loc);
  if (locWalkResult.wasInterrupted())
    return WalkResult::interrupt();
  if (locWalkResult.wasSkipped())
    return WalkResult::advance();

  return TypeSwitch<Location, WalkResult>(loc)
      .Case([&](mlir::CallSiteLoc callLoc) -> WalkResult {
        LocationAttr firstChoice, secondChoice;
        switch (policy) {
        case LocWalkPolicy::CalleePriority:
          secondChoice = callLoc.getCaller();
          firstChoice = callLoc.getCallee();
          break;
        case LocWalkPolicy::CallerPriority:
          secondChoice = callLoc.getCallee();
          firstChoice = callLoc.getCaller();
        }
        if (walkLocation(firstChoice, policy, walkFn).wasInterrupted())
          return WalkResult::interrupt();
        return walkLocation(secondChoice, policy, walkFn);
      })
      .Case([&](FusedLoc fusedLoc) -> WalkResult {
        for (Location subLoc : fusedLoc.getLocations())
          if (walkLocation(subLoc, policy, walkFn).wasInterrupted())
            return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .Case([&](mlir::NameLoc nameLoc) -> WalkResult {
        return walkLocation(nameLoc.getChildLoc(), policy, walkFn);
      })
      .Case([&](mlir::OpaqueLoc opaqueLoc) -> WalkResult {
        return walkLocation(opaqueLoc.getFallbackLocation(), policy, walkFn);
      })
      .Default(WalkResult::advance());
}

WalkResult DebugInfo::walkScope(Location loc, LocWalkPolicy policy,
                                function_ref<WalkResult(DIScopeAttr)> walkFn) {
  return walkLocation(loc, policy, [&](Location loc) {
    if (auto fusedLoc = dyn_cast<mlir::FusedLocWith<DIScopeAttr>>(loc))
      return walkFn(fusedLoc.getMetadata()).wasInterrupted()
                 ? WalkResult::interrupt()
                 : WalkResult::skip();
    return WalkResult::advance();
  });
}

FileLineColLoc DebugInfo::extractSourceLoc(Location callLoc) {
  FileLineColLoc resolvedLoc;
  DebugInfo::walkLocation(
      callLoc, DebugInfo::LocWalkPolicy::CalleePriority, [&](Location loc) {
        if (auto fileLineCol = dyn_cast<FileLineColLoc>(loc)) {
          resolvedLoc = fileLineCol;
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      });
  return resolvedLoc;
}

Location DebugInfo::stripDebugScopesRecursively(Location loc) {
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement(
      [&](mlir::FusedLocWith<DIAttr> diLoc) -> LocationAttr {
        ArrayRef<Location> locs = diLoc.getLocations();
        if (locs.size() == 1)
          return locs[0];
        return FusedLoc::get(diLoc.getContext(), locs);
      });
  return cast<LocationAttr>(replacer.replace(loc));
}

DISubprogramAttr DebugInfo::extractScope(mlir::FunctionOpInterface funcOp) {
  if (auto fusedLoc =
          dyn_cast<mlir::FusedLocWith<DISubprogramAttr>>(funcOp->getLoc()))
    return fusedLoc.getMetadata();
  return {};
}

DIScopeAttr DebugInfo::extractScope(Operation *op) {
  if (auto scopedOp = dyn_cast<DebugInfo::ScopedLocation>(op))
    return scopedOp.getLocScope();

  // For other ops, we look for the scope recursively.
  return extractScopeFrom<DIScopeAttr>(op->getLoc(),
                                       LocWalkPolicy::CalleePriority);
}

void DIAttrTypeReplacer::replaceElementsIn(Operation *op) {
  // As an optimization, we only replace attributes within the dictionaries of
  // DebugInfo operations. For everything else, we only check the location for
  // debug info.
  bool updateAttrs =
      llvm::isa_and_present<DebugInfo::DebugInfoDialect>(op->getDialect());
  AttrTypeReplacer::replaceElementsIn(op, updateAttrs, /*replaceLocs=*/true);
}

void DIAttrTypeReplacer::recursivelyReplaceElementsIn(Operation *op) {
  op->walk([&](Operation *op) { replaceElementsIn(op); });
}

void DebugInfo::updateSubprogram(mlir::FunctionOpInterface funcOp,
                                 StringAttr linkageName,
                                 SourceNameAttr sourceName) {
  DISubprogramAttr funcSp = extractScope(funcOp);
  if (!funcSp)
    return;

  if (!sourceName)
    sourceName = funcSp.getSourceName();
  DISubprogramAttr newAttr = funcSp.cloneWith(sourceName, linkageName);

  DIAttrTypeReplacer replacer;
  replacer.addReplacement(
      [&](DISubprogramAttr sp) { return sp == funcSp ? newAttr : sp; });
  replacer.recursivelyReplaceElementsIn(funcOp);
}

void DebugInfo::updateInlinedLoc(Operation *op, Location callerLoc) {
  if (auto inlined = dyn_cast<DebugInfo::InlinedSubprogramScoped>(op)) {
    if (LocationAttr callLoc = inlined.getCallLocAttr())
      inlined.setCallLocAttr(mlir::CallSiteLoc::get(callLoc, callerLoc));
    else
      inlined.setCallLocAttr(callerLoc);
  } else if (!isa<DebugInfo::SubprogramScoped>(op)) {
    if (op->hasTrait<OpTrait::ConstantLike>()) {
      // Workaround to handle CSE hoisting constants out of
      // InlinedSubprogramScoped ops into an outer scope and causing debug
      // scope mismatch (Tracker: https://linear.app/modularml/issue/MOCO-143).
      op->setLoc(stripDebugScopesRecursively(op->getLoc()));
    } else {
      op->setLoc(mlir::CallSiteLoc::get(op->getLoc(), callerLoc));
    }
  }
}
