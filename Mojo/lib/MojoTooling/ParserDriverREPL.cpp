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
//
// This file provides the main entrypoints for the Mojo parser.
//
//===----------------------------------------------------------------------===//

#include <cstdlib>

#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Mojo/MojoTooling/CodeComplete.h"
#include "Mojo/MojoTooling/PublicASTDecl.h"
#include "Mojo/POPDialect/POPOps.h"
#include "ParserDriverImpl.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// MojoParserContext::REPLLocMapper
//===----------------------------------------------------------------------===//

/// This class provides support for mapping locations between an input REPL
/// expression and the wrapped expression that is actually parsed.
class MojoParserContext::REPLLocMapper::ExprLocMapper {
public:
  ExprLocMapper(StringRef inputExpr)
      : inputExpr(inputExpr), inputToWrappedMap(allocator),
        wrappedToInputMap(allocator) {}

  /// Set the expressions mapped in this mapper.
  void setWrappedExpr(StringRef exprText) { wrappedExpr = exprText; }

  /// Map a substring of the REPL input expression to the same corresponding
  /// substring within the wrapped expression.
  void addMapping(StringRef inputExprSplice, unsigned wrappedExprOffset) {
    // Insert a mapping from input to wrapped expression.
    unsigned inputExprOffset = inputExprSplice.data() - inputExpr.data();
    inputToWrappedMap.insert(inputExprOffset,
                             inputExprOffset + inputExprSplice.size(),
                             wrappedExprOffset);

    // Insert a reverse mapping from wrapped to input expression.
    wrappedToInputMap.insert(wrappedExprOffset,
                             wrappedExprOffset + inputExprSplice.size(),
                             inputExprOffset);
  }

  /// Map the given location in the input expression to the wrapped expression.
  /// Returns an invalid location if the location is not mapped.
  llvm::SMLoc mapLocation(llvm::SMLoc loc) const {
    auto mapImpl = [&](const char *locBufferStart, const char *newBufferStart,
                       const MapT &map) {
      unsigned locOffset = loc.getPointer() - locBufferStart;

      auto it = map.find(locOffset);
      if (!it.valid() || locOffset < it.start())
        return llvm::SMLoc();
      return llvm::SMLoc::getFromPointer(newBufferStart + it.value() +
                                         (locOffset - it.start()));
    };

    // Check if the location is within the input or wrapped expression.
    if (loc.getPointer() >= inputExpr.data() &&
        loc.getPointer() < inputExpr.end()) {
      return mapImpl(inputExpr.data(), wrappedExpr.data(), inputToWrappedMap);
    }
    if (loc.getPointer() >= wrappedExpr.data() &&
        loc.getPointer() < wrappedExpr.end()) {
      return mapImpl(wrappedExpr.data(), inputExpr.data(), wrappedToInputMap);
    }
    return llvm::SMLoc();
  }

private:
  using MapT = llvm::IntervalMap<
      unsigned, unsigned,
      llvm::IntervalMapImpl::NodeSizer<unsigned, StringRef>::LeafSize,
      llvm::IntervalMapHalfOpenInfo<unsigned>>;
  MapT::Allocator allocator;

  /// The buffer for the input expression.
  StringRef inputExpr;
  MapT inputToWrappedMap;

  /// The buffer for the wrapped expression.
  StringRef wrappedExpr;
  MapT wrappedToInputMap;
};

MojoParserContext::REPLLocMapper::REPLLocMapper(llvm::SourceMgr &sourceMgr)
    : sourceMgr(sourceMgr) {}
MojoParserContext::REPLLocMapper::~REPLLocMapper() = default;

llvm::SMLoc
MojoParserContext::REPLLocMapper::mapLocation(llvm::SMLoc loc) const {
  for (ExprLocMapper &mapper : llvm::make_pointee_range(exprMappers))
    if (llvm::SMLoc newLoc = mapper.mapLocation(loc); newLoc.isValid())
      return newLoc;

  return llvm::SMLoc();
}

llvm::SMRange
MojoParserContext::REPLLocMapper::mapRange(llvm::SMRange range) const {
  SMLoc newStart = mapLocation(range.Start);
  SMLoc newEnd = mapLocation(range.End);
  if (newStart.isValid() && !newEnd.isValid()) {
    // If we are in an exclusive end that couldn't be mapped, we map the
    // previous location (inclusive) and then adjust the offset.
    if (!newEnd.isValid() && range.End.getPointer()) {
      newEnd = mapLocation(SMLoc::getFromPointer(range.End.getPointer() - 1));
      if (newEnd.isValid())
        newEnd = SMLoc::getFromPointer(newEnd.getPointer() + 1);
    }
  }
  return llvm::SMRange(newStart, newEnd);
}

llvm::SMDiagnostic MojoParserContext::REPLLocMapper::mapDiagnostic(
    const llvm::SMDiagnostic &diag) {
  // Check if the diagnostic is using location information from the wrapped
  // expression.
  llvm::SMLoc newLoc = mapLocation(diag.getLoc());
  if (!newLoc.isValid())
    return diag;

  // If we remapped the location back to the input, we need to update the
  // components of the diagnostic to account for the new location information.
  auto [newLine, newCol] = sourceMgr.getLineAndColumn(newLoc);
  --newCol;
  int colDiff = diag.getColumnNo() - newCol;

  // Update the diagnostic contents based on the column difference.
  SmallVector<std::pair<unsigned, unsigned>> ranges(diag.getRanges());
  std::string lineContents = diag.getLineContents().str();
  if (colDiff) {
    for (auto &range : ranges) {
      range.first -= colDiff;
      range.second -= colDiff;
    }
    if (!lineContents.empty()) {
      if (colDiff > 0)
        lineContents.erase(0, colDiff);
      else
        lineContents.insert(0, -colDiff, ' ');
    }
  }

  // Update the locations of the fixits.
  SmallVector<llvm::SMFixIt> fixits;
  for (auto &fixit : diag.getFixIts()) {
    // Only include the fix-it if its range can be mapped back; an unmapped
    // fix-it cannot be presented to the user and would crash SMFixIt.
    auto mappedRange = mapRange(fixit.getRange());
    if (mappedRange.isValid())
      fixits.emplace_back(mappedRange, fixit.getText());
  }

  // Remap the file name and record the diagnostic.
  StringRef newFileName =
      sourceMgr.getMemoryBuffer(sourceMgr.FindBufferContainingLoc(newLoc))
          ->getBufferIdentifier();
  return llvm::SMDiagnostic(sourceMgr, newLoc, newFileName, newLine, newCol,
                            diag.getKind(), diag.getMessage(), lineContents,
                            ranges, fixits);
}

//===----------------------------------------------------------------------===//
// Expression Extraction
//===----------------------------------------------------------------------===//

/// Return the indentation level of the first line of the string.
static size_t getIndentationLevel(StringRef str) {
  return str.size() - str.ltrim().size();
}

/// Return true if the given line matches any of the given prefixes.
template <typename Prefixes>
static bool matchesAnyPrefix(StringRef line, const Prefixes &prefixes) {
  return llvm::any_of(
      prefixes, [&](StringRef prefix) { return line.starts_with(prefix); });
}

static bool isFunctionOrStructDeclaration(StringRef code) {
  static constexpr auto kPrefixes = {
      "fn ",
      "def ",
      "struct ",
      "trait ",
  };
  return matchesAnyPrefix(code, kPrefixes);
}

static bool isDecorator(StringRef code) { return code.starts_with("@"); }

static bool isIndented(StringRef code) {
  static constexpr auto kPrefixes = {" ", "\t"};
  return matchesAnyPrefix(code, kPrefixes);
}

static bool isSimpleImport(StringRef code) {
  // `import` is a reserved keyword.
  return code.starts_with("import ");
}

static bool isFromImport(StringRef code) {
  // `from` is a reserved keyword.
  return code.starts_with("from ");
}

static bool isAlias(StringRef code) {
  return code.starts_with("alias ") || code.starts_with("comptime");
}

static bool isOpenParenthesis(char c) { return c == '(' || c == '['; }

static bool isCloseParenthesis(char c) { return c == ')' || c == ']'; }

/// Parse the beginning of `unparsedCode` as a simple `import *` statement. If
/// the parsing fails, false is returned. `unparsedCode` is modified to point to
/// the next statement is the parsing was successful, in which case true is
/// returned.
static bool tryHandleSimpleImport(unsigned &line, unsigned lineE,
                                  ArrayRef<StringRef> exprLines,
                                  SmallVectorImpl<StringRef> &topLevelCode) {
  if (!isSimpleImport(exprLines[line]))
    return false;
  topLevelCode.push_back(exprLines[line++]);
  return true;
}

/// Parse the beginning of `unparsedCode` as a `from * import` statement, a
/// `fn`, a `def` or a `struct` top level statement. If the parsing fails, false
/// is returned. `unparsedCode` is modified to point to the next statement is
/// the parsing was successful, in which case true is returned.
static bool tryHandleFromImportAliasFunctionOrStruct(
    unsigned &line, unsigned lineE, ArrayRef<StringRef> exprLines,
    SmallVectorImpl<StringRef> &topLevelCode) {
  // A decorator line (@fieldwise_init, @always_inline, etc.) is pushed to
  // topLevelCode as a single line so it stays adjacent to the fn/struct
  // declaration that follows it, which will be processed on the next iteration.
  // Body absorption and colon-searching belong to the declaration line, not
  // the decorator.
  if (isDecorator(exprLines[line])) {
    topLevelCode.push_back(exprLines[line++]);
    return true;
  }
  bool isFunctionOrStruct = isFunctionOrStructDeclaration(exprLines[line]);
  if (!isFunctionOrStruct && !isFromImport(exprLines[line]) &&
      !isAlias(exprLines[line]))
    return false;

  // These statements can have a hierarchy of () or [], so we need to parse
  // until we have visited all of them.

  // If we are in a function or struct, we also need to find a : outside of any
  // parenthesis.
  bool requiresOuterColon = isFunctionOrStruct;

  // The following block will find the top declaration and not the body of the
  // entity we are parsing. For example, if we have the function
  //
  //   def foo() -> Int:
  //     return 12
  //
  // then this block find the `def foo() -> Int:\n`, even if it's split across
  // many lines. The body will be handled later.
  // Traverse the body searching for the end of the signature, making sure to
  // properly match groups of ( or [.
  for (size_t openings = 0; line < lineE;) {
    topLevelCode.push_back(exprLines[line++]);
    for (char c : topLevelCode.back()) {
      // Skip past comments.
      if (c == '#')
        break;

      if (isOpenParenthesis(c)) {
        ++openings;
      } else if (isCloseParenthesis(c)) {
        --openings;
      } else if (c == ':' && openings == 0) {
        requiresOuterColon = false;
        break;
      }
    }
    if (openings == 0 && !requiresOuterColon)
      break;
  }

  if (isFunctionOrStruct) {
    // We now absorb all indented code including empty lines, which make the
    // body of the entity we are parsing. This doesn't apply to aliases, for
    // example.
    for (; line < lineE; ++line) {
      StringRef lineStr = exprLines[line];
      if (!lineStr.empty() && !isIndented(lineStr) && !lineStr.starts_with("#"))
        break;
      topLevelCode.push_back(lineStr);
    }
  }
  return true;
}

static void extractExpressionCode(StringRef exprText,
                                  SmallVectorImpl<StringRef> &topLevelCode,
                                  SmallVectorImpl<StringRef> &mainBodyCode) {
  SmallVector<StringRef> exprLines;
  exprText.split(exprLines, "\n");

  // Determine the minimum indentation of the expression.
  size_t indent = std::numeric_limits<size_t>::max();
  for (StringRef &line : exprLines) {
    // Trim out any carriage returns.
    line = line.trim("\r");
    if (line.empty())
      continue;

    indent = std::min(indent, getIndentationLevel(line));
    if (indent == 0)
      break;
  }

  for (StringRef &line : exprLines) {
    if (line.empty())
      continue;
    // Remove the necessary indentation from the expression lines.
    if (indent && indent != std::numeric_limits<size_t>::max())
      line = line.drop_front(indent);

    // Handle the REPL magic that is used to control code showed as
    // documentation. We handle it here to preserve the indentation/whitespace
    // for location information. Handle empty hidden lines as well, by
    // accounting for the magic indentation being optional.
    if (line.consume_front("%#"))
      line.consume_front(" ");
  }

  // The following code will consume chunks of code assigning them to either
  // the top-level or the main body sections.
  for (unsigned line = 0, lineE = exprLines.size(); line < lineE;) {
    // Note: We are not yet handling multiline expressions with \.
    if (!tryHandleFromImportAliasFunctionOrStruct(line, lineE, exprLines,
                                                  topLevelCode) &&
        !tryHandleSimpleImport(line, lineE, exprLines, topLevelCode)) {
      // Any other case is just main body code.
      if (!exprLines[line].empty())
        mainBodyCode.push_back(exprLines[line]);
      ++line;
    }
  }
}

//===----------------------------------------------------------------------===//
// Expression Wrapping
//===----------------------------------------------------------------------===//

/// This code block represents a marker used to split the top-level code of an
/// expression from the wrapped entry point functions.
const char *kEntryPointEndMarker = "## End Wrapped Mojo REPL EntryPoint ##\n";

/// Return a unique type name for a persistent variable.
static std::string getPersistentVariableTypeName(StringRef name) {
  return llvm::formatv("__mojo_repl_persistent_var_type_{0}", name).str();
}

/// Wrap the provided expression text in a function so that it can be executed.
/// The generated function uses the provided name, and the provided variables
/// are passed via fields to a generated struct that is used as the first
/// argument of the function.
static std::string
wrapExpressionText(MojoParserContext::REPLLocMapper::ExprLocMapper &locMapper,
                   StringRef wrappedFnName, StringRef exprText,
                   ArrayRef<std::pair<StringRef, Type>> variables,
                   bool isFirstREPLCell) {
  // Wrap the expression text in a function so that we can execute it.
  std::string transformedText;
  llvm::raw_string_ostream exprOS(transformedText);

  // Insert a preamble of imports used by the expression wrapper.
  if (isFirstREPLCell) {
    exprOS << "from std.memory import Pointer as "
           << "__mojo_repl_UnsafePointer\n"
           << "from std.python.python import Python as __mojo_repl_Python\n"
           << "from std.memory import UnsafePointer\n";
  }

  // Extract out the top-level code from the expression code.
  SmallVector<StringRef> topLevelCode, mainBodyCode;
  extractExpressionCode(exprText, topLevelCode, mainBodyCode);

  // Build a mapping for pieces of the input expression and the wrapped
  // expression, enabling seamless location mapping between the two.
  auto emitAndMapCode = [&](StringRef code) {
    if (!code.empty())
      // Map the code and the '\n' appended below (+1) so that positions just
      // past the last token on a line (e.g. insertAfterToken fix-its) can be
      // mapped back to the input.  addMapping only uses the size for interval
      // arithmetic and never dereferences the bytes, so size+1 is safe even
      // when there is no '\n' in the input at code.end().
      locMapper.addMapping(StringRef(code.data(), code.size() + 1),
                           exprOS.str().size());
    exprOS << code << "\n";
  };

  // Build the input struct, which contains each of the persistent variables.
  exprOS << "struct __mojo_repl_context__(Movable where False):\n";
  for (auto &[name, type] : variables) {
    exprOS << llvm::formatv("  @__allow_legacy_any_origin_fields\n"
                            "  var `{0}`: "
                            "__mojo_repl_UnsafePointer[mut=True, "
                            "__mojo_repl_UnsafePointer[mut=True, {1}, "
                            "MutAnyOrigin], MutAnyOrigin]\n",
                            name, getPersistentVariableTypeName(name));
  }
  if (variables.empty())
    exprOS << "  pass\n";
  exprOS << "\n";

  // Generate a wrapper function to handle the extracting function arguments as
  // references.
  exprOS << "def " << wrappedFnName
         << "(mut __mojo_repl_arg: __mojo_repl_context__):\n"
            "  try:\n"
            "    __mojo_repl_expr_impl__(__mojo_repl_arg";
  for (auto &[name, type] : variables)
    exprOS << formatv(", __mojo_repl_arg.`{0}`[][]", name);

  exprOS << ")\n"
            "  except error:\n"
            "    print(\"Error:\", error)\n\n";

  // Finally we can generate the actual expression function.
  exprOS << "def __mojo_repl_expr_impl__(mut __mojo_repl_arg: "
            "__mojo_repl_context__";
  for (auto &[name, type] : variables)
    exprOS << llvm::formatv(", mut `{0}`: {1}", name,
                            getPersistentVariableTypeName(name));
  exprOS << ") raises -> None:\n";

  // Splat out the main body code inside of a nested def. This will allow for us
  // to redefine previous variables transparently.
  exprOS << "  var __mojo_repl_expr_failed = True\n"
            "  @__parameter\n"
            "  def __mojo_repl_expr_body__() raises -> None:\n";

  // The following is the other chunk of code just written by the user.
  for (StringRef code : mainBodyCode) {
    exprOS << "    ";
    emitAndMapCode(code);
  }
  exprOS << "    pass\n";

  // If the code succeeded, reset the failure flag.
  exprOS << "  __mojo_repl_expr_body__()\n"
            "  __mojo_repl_expr_failed = False\n";

  // Emit a marker separating the top-level code from the forthcoming entry
  // point logic.
  exprOS << kEntryPointEndMarker;

  // Splat out the top-level code.
  for (StringRef code : topLevelCode)
    emitAndMapCode(code);

  return exprOS.str();
}

//===----------------------------------------------------------------------===//
// Persistent Variables
//===----------------------------------------------------------------------===//

// Simple utility functor for looking up a decl that's known to exist.
static ASTDecl &lookupSingleDecl(ASTDecl &decl, StringRef name) {
  return *decl.lookupInCurrentScope(name).front();
}

/// Process all of the top-level variables defined within the expression body to
/// see which should be persisted. If a variable is persisted, it will be
/// added to the state struct and the expression body will be rewritten to
/// access it via the state struct.
/// TODO: It'd be a bit nicer to have this handled when actually parsing the
/// variables, but for now we do this as a post-processing step.
static void processVariablesForPersistence(MojoParserREPLListener &listener,
                                           ASTDecl &exprFnDecl,
                                           ASTDecl &stateStructDecl) {
  auto exprFn = cast<FnOp>(exprFnDecl.getIfOperation());
  auto stateStruct = cast<StructDeclOp>(stateStructDecl.getIfOperation());

  // Grab all of the variables within the expression body and sort them by name,
  // so that we can deterministically process them.
  SmallVector<std::pair<StringAttr, ASTDecl *>> variables;
  auto addVars = [&](ASTDecl &decl) {
    for (auto &[name, decls] : decl.getDeclsInScope())
      if (decls.size() == 1 &&
          isa_and_nonnull<VarDeclOp>(decls.front()->getIfOperation()))
        variables.emplace_back(name, decls.front());
  };
  addVars(exprFnDecl);
  addVars(lookupSingleDecl(exprFnDecl, "__mojo_repl_expr_body__"));

  llvm::sort(variables, [](const auto &lhs, const auto &rhs) {
    return lhs.first.getValue() < rhs.first.getValue();
  });

  OpBuilder structBuilder = OpBuilder::atBlockEnd(stateStruct.getBody());
  Value structValue = exprFn.getArgument(0);
  TypedAttr targetAttr = ParamOperatorAttr::get(
      POC::CurrentTarget, /*operands=*/{}, structBuilder.getType<TargetType>());

  // Utility functor to check if a variable should be inserted, and if so insert
  // a new field into the persistent state struct. If the variable was
  // persisted, returns a value corresponding to the address of the field.
  // Returns nullptr otherwise.
  auto anyRegTypeType = structBuilder.getType<TypeType>();
  auto checkInsertPersistentVar = [&](VarDeclOp varOp) -> MRValue {
    Type elementType = varOp.getType().getElementType();
    PointerType type = PointerType::get(elementType);

    // Check if the variable should be persisted.
    if (!listener.shouldPersistVariable(varOp.getNameAttr(), elementType))
      return {};

    // The variable was persisted, insert a new field into the state struct.
    std::string newFieldName =
        ("__new_repl_var_" + varOp.getNameAttr().strref()).str();
    auto newField = LIT::StructFieldOp::create(
        structBuilder, varOp->getLoc(), newFieldName, PointerType::get(type));

    // Materialize a reference to the variable within the function.
    ImplicitLocOpBuilder builder(varOp->getLoc(), varOp);
    Value fieldGep = LIT::RefStructGEROp::create(builder, varOp->getLoc(),
                                                 structValue, newField);
    Value fieldLoad = RefLoadOp::create(builder, varOp->getLoc(), fieldGep);

    // TODO: Whenever we have globals, we should be able to use a global
    // variable for the address and ensure it gets preserved. For now, we just
    // malloc the memory.
    SmallVector<TypedAttr> operands{
        TypeParamAttr::get(elementType, anyRegTypeType), targetAttr};
    // Compute the size of the type.
    Value sizeOf = ParamConstantOp::create(
        builder, ParamOperatorAttr::get(POC::GetSizeOf, operands));
    // Compute the alignment of the type.
    Value alignOf = ParamConstantOp::create(
        builder, ParamOperatorAttr::get(POC::GetAlignOf, operands));
    // Allocate an aligned blob for the variable.
    Value mallocCast = POP::AlignedAllocOp::create(
        builder, type, ArrayRef<Value>{alignOf, sizeOf});
    POP::StoreOp::create(builder, mallocCast, fieldLoad);

    // Return a pointer to the new address of the variable.

    // In order to use the pointer as a reference we force cast.
    // FIXME(references): switch AlignedAllocOp to use references

    // Declare the origin as a placeholder, we're going to replace this,
    // so we need to define the origin.
    TypedAttr origin = varOp.getType().getOrigin();
    ParamDeclareOp::create(builder, varOp.getParamDecl(),
                           AnyOriginAttr::get(origin.getType()));

    // Create an untracked reference from this.
    // FIXME: this should really use RefFromPointerOp to make sure that
    // CheckLifetimes is enagaged.  Unfortunately the way the REPL is modeling
    // this breaks throwing functions, because CheckLifetimes (correctly)
    // detects that the reference doesn't get initialized if a function throws.
    //
    // Workaround this for now by using hacky RefFromPointerREPLOp.
    // NOTE: DO NOT ADOPT THIS ANYWHERE ELSE!
    mallocCast = LIT::RefFromPointerREPLOp::create(
        builder, varOp.getType(), mallocCast, varOp.getNameAttr());
    return MRValue(mallocCast);
  };

  for (auto &[name, decl] : variables) {
    // Handle memory based decls.
    if (auto varOp = dyn_cast_or_null<LIT::VarDeclOp>(decl->getIfOperation())) {
      if (MRValue field = checkInsertPersistentVar(varOp)) {
        varOp.replaceAllUsesWith(field);
        varOp.erase();
        decl->setIRValue(field);
      }
      continue;
    }
  }
}

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

namespace {
/// This class implements a diagnostic handler for REPL cells.
class REPLDiagnosticHandler {
public:
  REPLDiagnosticHandler(MojoParserREPLListener &listener,
                        MojoParserContext::REPLLocMapper &locMapper,
                        StringRef exprText, llvm::SourceMgr &sourceMgr)
      : listener(listener), locMapper(locMapper), exprText(exprText) {
    sourceMgr.setDiagHandler(handleDiagnostic, this);
  }

  /// This method processes all of the diagnostics that have been collected.
  /// `exprText` is the text of the wrapped expression that was parsed.
  LogicalResult processDiagnostics();

private:
  /// A static diagnostic handler function that is usable with SourceMgr. This
  /// handler simply collects diagnostics, which will get processed later.
  static void handleDiagnostic(const llvm::SMDiagnostic &diagnostic,
                               void *ctx) {
    auto *handler = static_cast<REPLDiagnosticHandler *>(ctx);
    handler->diagnostics.emplace_back(
        handler->locMapper.mapDiagnostic(diagnostic));
  }

  MojoParserREPLListener &listener;
  MojoParserContext::REPLLocMapper &locMapper;
  StringRef exprText;
  std::vector<llvm::SMDiagnostic> diagnostics;
};
} // namespace

LogicalResult REPLDiagnosticHandler::processDiagnostics() {
  if (diagnostics.empty())
    return success();

  // Notify the listener of the diagnostics.
  listener.notifyDiagnostics(diagnostics);

  // Process all of the diagnostics to check for errors, and apply fixits if
  // possible.

  // This takes advantage of the fact that fixits are ordered to apply multiple
  // fixits to a single expression.
  std::string newText;
  size_t prevEnd = 0;
  auto applyFixit = [&](const llvm::SMFixIt &fixit) -> LogicalResult {
    llvm::SMRange range = fixit.getRange();
    if (!range.isValid())
      return failure();

    StringRef removedText(range.Start.getPointer(),
                          range.End.getPointer() - range.Start.getPointer());
    StringRef insertedText = fixit.getText();

    // The current range starts at the previous end pointer.
    StringRef currentOriginalRange(exprText.begin() + prevEnd);

    // Add the substring from the start of the current original text range.
    if (range.Start.getPointer() < currentOriginalRange.end() &&
        range.Start.getPointer() >= currentOriginalRange.begin())
      newText += currentOriginalRange.substr(
          0, range.Start.getPointer() - currentOriginalRange.begin());

    // Add the text to insert.
    newText += insertedText;

    // Update prevEnd. At the *very* end, we will clean up by adding the
    // remaining substring. Subtract off the size of the inserted text because
    // the pointers are all indexed off the original text.
    prevEnd += range.End.getPointer() - currentOriginalRange.begin();
    return success();
  };

  bool hadFixit = false;
  bool allDiagsHandled = llvm::all_of(diagnostics, [&](const auto &diag) {
    if (diag.getFixIts().empty())
      return diag.getKind() != llvm::SourceMgr::DK_Error;

    hadFixit = true;
    return llvm::all_of(diag.getFixIts(), [&](const llvm::SMFixIt &fixit) {
      return succeeded(applyFixit(fixit));
    });
  });

  // If we handled all the diagnostics and we applied fixits, notify the
  // listener that we have an improved expression.
  if (allDiagsHandled && hadFixit) {
    // Complete fixit handling by adding the substring from prevEnd to the end
    // of the buffer. We do this here because we only want to do it if/once
    // *all* diagnostics are handled.
    newText += exprText.substr(prevEnd);
    listener.notifyFixedExpr(newText);
  }

  return success(allDiagsHandled);
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

/// Build a module decl for use in a REPL expression.
static ASTDecl &buildREPLModule(const llvm::MemoryBuffer *sourceBuf,
                                StringRef moduleName,
                                SharedState &sharedState) {
  StringRef exprId = sourceBuf->getBufferIdentifier();

  // If we are emitting debug info, create a file entry for this file.
  DebugInfo::DIBuilder::ScopeGuard fileGuard;
  if (sharedState.diBuilder)
    fileGuard = sharedState.diBuilder->pushFile(exprId);

  // Create the input module.
  MLIRContext *ctx = sharedState.getContext();
  auto fileLoc = FileLineColLoc::get(ctx, exprId, /*line=*/1, /*column=*/1);
  ASTDecl &decl = sharedState.createModule(moduleName, sourceBuf, fileLoc);
  (void)sharedState.declResolver->resolveBody(decl, decl.getLoc());
  return decl;
}

/// Build and resolve a REPL module for the given wrapped expression string.
/// Returns the fully resolved REPL module decl.
static ASTDecl &buildAndResolveREPLModule(
    const llvm::MemoryBuffer *sourceBuf, StringRef moduleName,
    SharedState &sharedState, MojoASTDeclRef prevReplExpr, bool parseForLSP,
    ArrayRef<std::pair<StringRef, Type>> replVariables = {}) {
  ASTDecl &moduleDecl = buildREPLModule(sourceBuf, moduleName, sharedState);

  // Generate aliases for the types of any persistent variables. We do this
  // programmatically because we can't guarantee we can print the type in a way
  // that the mojo parser will accept.
  for (auto [name, type] : replVariables) {
    // The persistent variable's type can contain emitted references to type
    // decls. We have to make sure to resolve them in the current context. We
    // can use the SharedState's type walker for this.
    (void)sharedState.resolveDeclReferencesIn(SMLoc(), type);

    std::string typeName = getPersistentVariableTypeName(name);
    PValue typeValue(type);

    OpBuilder builder = moduleDecl.getDeclEndBuilder();
    AliasDeclOp typeDecl = AliasDeclOp::create(
        builder, builder.getUnknownLoc(),
        ParamDeclAttr::get(typeName, typeValue.getType()), typeValue.get());
    sharedState.declResolver->addFullyResolvedDecl(&*typeDecl, typeName,
                                                   SMLoc(), &moduleDecl);
  }

  // Before resolving everything in the REPL cell, resolve the body and import
  // as many of the previously defined REPL decls that we can.
  if (prevReplExpr) {
    // Explicitly import any decls from the previous REPL module that aren't
    // already defined in the current module. We can't use wildcards here
    // because we also want to import _ and other traditionally "hidden" decls
    // from previous cells.
    SmallVector<std::pair<StringAttr, const TinyPtrVector<ASTDecl *>>> fnDecls;
    for (auto &[name, decls] : prevReplExpr->getDeclsInScope()) {
      auto existingDecls = moduleDecl.lookupInCurrentScope(name);
      if (existingDecls.empty()) {
        sharedState.declResolver->aliasDecls(decls, name, SMLoc(), moduleDecl);
        continue;
      }
      // If we hit an overlap and these are function decls, save them for
      // processing for later. We might be able to import if the signatures
      // don't overlap.
      if (isa_and_nonnull<FnOp>(existingDecls.front()->getIfOperation()) &&
          isa_and_nonnull<FnOp>(decls.front()->getIfOperation())) {
        fnDecls.push_back({name, decls});
      }
    }

    // Now that we've imported all of the decls we can, go ahead and import the
    // functions that have name overlaps. We do this afterwards so that we can
    // resolve the signature of the pre-existing functions to see if there are
    // signature overlaps (to avoid duplicate function declarations).
    for (auto &[name, decls] : fnDecls) {
      (void)sharedState.declResolver->tryAliasDecls(decls, name, SMLoc(),
                                                    moduleDecl);
    }
  }

  // With the top-level of the file parsed, resolve all deferred declarations.
  // The LSP doc-string path (parseForLSP=true) only needs
  // signatures of library functions referenced from the code block: it uses
  // resolveForLSP to body-resolve the direct children, then
  // resolveSignaturesForLSP to signature-resolve transitive deps. All other
  // paths (interactive REPL, notebook cells, LLDB) need full body resolution
  // because they compile and execute the generated code.
  if (parseForLSP) {
    resolveForLSP(*sharedState.declResolver, moduleDecl);
    if (!sharedState.diags.isErrorEmitted())
      resolveSignaturesForLSP(*sharedState.declResolver);
  } else {
    // Keep unparsed decls alive — later cells may reference them.
    sharedState.declResolver->resolveAllReferencedFrom(
        moduleDecl, /*eraseUnparsedDecls=*/false);
  }

  // Resolve any imported wildcard decls, this ensures those decls will be
  // available for future cells.
  (void)sharedState.declResolver->resolveAllWildcardImports(moduleDecl);
  return moduleDecl;
}

MojoParserContext::REPLLocMapper &MojoParserContext::getREPLLocMapper() {
  return impl->replLocMapper;
}

MojoParserContext::ParsedREPLExpr MojoParserContext::parseREPLExpression(
    MojoParserREPLListener &listener, unsigned exprFileId,
    StringRef replExprFnName,
    ArrayRef<std::pair<StringRef, Type>> replVariables) {
  MojoASTDeclRef prevReplExpr;
  if (!impl->replModuleDecls.empty())
    prevReplExpr = impl->replModuleDecls.back();

  llvm::SourceMgr &sourceMgr = getSourceMgr();
  const llvm::MemoryBuffer *exprFileBuf = sourceMgr.getMemoryBuffer(exprFileId);
  return parseREPLExpression(listener, exprFileId, exprFileBuf->getBuffer(),
                             replExprFnName, replVariables, prevReplExpr,
                             /*parseForLSP=*/false);
}

MojoParserContext::ParsedREPLExpr MojoParserContext::parseREPLExpression(
    MojoParserREPLListener &listener, unsigned exprFileId, StringRef exprText,
    StringRef replExprFnName,
    ArrayRef<std::pair<StringRef, Type>> replVariables,
    MojoASTDeclRef prevReplExpr, bool parseForLSP) {
  llvm::SourceMgr &sourceMgr = getSourceMgr();
  const llvm::MemoryBuffer *exprFileBuf = sourceMgr.getMemoryBuffer(exprFileId);
  assert(exprFileBuf->getBufferStart() <= exprText.data() &&
         exprFileBuf->getBufferEnd() >= exprText.data() + exprText.size() &&
         "expected exprText to be a substring of exprFileBuf");

  // Build a location mapper for this expression.
  impl->replLocMapper.exprMappers.emplace_back(
      std::make_unique<REPLLocMapper::ExprLocMapper>(exprText));
  REPLLocMapper::ExprLocMapper &exprLocMapper =
      *impl->replLocMapper.exprMappers.back();

  // Set up a diagnostic handler to process diagnostics emitted during parsing.
  auto oldDiagHandler = sourceMgr.getDiagHandler();
  auto oldDiagContext = sourceMgr.getDiagContext();
  auto resetHandlerOnExit = llvm::scope_exit(
      [&] { sourceMgr.setDiagHandler(oldDiagHandler, oldDiagContext); });
  REPLDiagnosticHandler diagHandler(listener, impl->replLocMapper, exprText,
                                    sourceMgr);

  // Wrap the expression text in a function so that we can execute it.
  std::string wrappedExprText =
      wrapExpressionText(exprLocMapper, replExprFnName, exprText, replVariables,
                         /*isFirstREPLCell=*/!prevReplExpr);
  listener.notifyWrappedExpr(wrappedExprText);

  // TODO: We should print the expression to a file if we need debug
  // information attached.
  size_t exprTextOffset = exprText.data() - exprFileBuf->getBufferStart();
  std::string replModuleName =
      (exprFileBuf->getBufferIdentifier() +
       (exprTextOffset ? (" wrapper_at(" + Twine(exprTextOffset) + ") ")
                       : Twine(" wrapper")))
          .str();
  auto buffer =
      llvm::MemoryBuffer::getMemBufferCopy(wrappedExprText, replModuleName);

  unsigned bufferId =
      sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  impl->sharedState.registerWrapperBuffer(bufferId,
                                          exprFileBuf->getBufferIdentifier());
  const llvm::MemoryBuffer *sourceBuf = sourceMgr.getMemoryBuffer(bufferId);

  exprLocMapper.setWrappedExpr(sourceBuf->getBuffer());

  // Resolve a module decl for this REPL expression.
  ASTDecl &moduleDecl =
      buildAndResolveREPLModule(sourceBuf, replModuleName, impl->sharedState,
                                prevReplExpr, parseForLSP, replVariables);
  if (prevReplExpr)
    impl->prevReplModuleDecls.insert({&moduleDecl, &*prevReplExpr});

  // Clear up the error state so that we are still able to parse future cells,
  // we'll handle diagnostic checks below.
  impl->sharedState.diags.clear();

  // Check if we have a non-recoverable parse error, or emitted an error and
  // then recovered.
  if (failed(diagHandler.processDiagnostics()))
    return {MojoASTDeclRef(&moduleDecl), MojoASTDeclRef()};

  // Process variables within the expression function for persistence.
  processVariablesForPersistence(
      listener, lookupSingleDecl(moduleDecl, "__mojo_repl_expr_impl__"),
      lookupSingleDecl(moduleDecl, "__mojo_repl_context__"));

  // Update the last REPL module decl.
  impl->replModuleDecls.push_back(&moduleDecl);
  ASTDecl *fnDecl = &lookupSingleDecl(moduleDecl, replExprFnName);
  return {MojoASTDeclRef(&moduleDecl), MojoASTDeclRef(fnDecl)};
}

void MojoParserContext::removeLastREPLExpression() {
  assert(!impl->replModuleDecls.empty() && "expected at least one REPL module");
  impl->replModuleDecls.pop_back();
}

bool MojoParserContext::isHiddenPersistentVariable(StringRef name) {
  return name.starts_with("__mojo_repl_expr");
}

//===----------------------------------------------------------------------===//
// Code Completion/Signature Help
//===----------------------------------------------------------------------===//

/// Collect the ordered chain of REPL modules that should be in the completion
/// cache. When replDecl is set (notebook editing a specific cell), we need
/// modules up to its predecessor; otherwise, all evaluated modules.
static SmallVector<ASTDecl *> collectCompletionTargetModules(
    ArrayRef<ASTDecl *> replModuleDecls,
    const llvm::MapVector<ASTDecl *, ASTDecl *> &prevReplModuleDecls,
    ASTDecl *replDecl) {
  SmallVector<ASTDecl *> targetModules;
  if (replDecl) {
    ASTDecl *decl = prevReplModuleDecls.lookup(replDecl);
    while (decl) {
      targetModules.push_back(decl);
      decl = prevReplModuleDecls.lookup(decl);
    }
    std::reverse(targetModules.begin(), targetModules.end());
  } else {
    targetModules.assign(replModuleDecls.begin(), replModuleDecls.end());
  }
  return targetModules;
}

/// Replay a single REPL module into a completion cache context. Extracts the
/// declarations (after the entry point marker) from the main SourceMgr and
/// parses them into the cache context.
static ASTDecl *replayModuleIntoCache(MojoParserContext &cacheCtx,
                                      llvm::SourceMgr &cacheSourceMgr,
                                      llvm::SourceMgr &mainSourceMgr,
                                      ASTDecl *module,
                                      MojoASTDeclRef prevDecl) {
  int bufferId = mainSourceMgr.FindBufferContainingLoc(module->getLoc());
  const llvm::MemoryBuffer *moduleBuf = mainSourceMgr.getMemoryBuffer(bufferId);
  StringRef moduleBufCode =
      moduleBuf->getBuffer().split(kEntryPointEndMarker).second;
  auto completionBuf = llvm::MemoryBuffer::getMemBuffer(
      moduleBufCode, moduleBuf->getBufferIdentifier());
  int completionBufferId =
      cacheSourceMgr.AddNewSourceBuffer(std::move(completionBuf), SMLoc());
  return &buildAndResolveREPLModule(
      cacheSourceMgr.getMemoryBuffer(completionBufferId),
      moduleBuf->getBufferIdentifier(), cacheCtx.getSharedState(), prevDecl,
      /*parseForLSP=*/false);
}

/// Prepare the wrapped expression text for completion, inserting and removing
/// the completion marker to track position through the wrapping transformation.
static std::string prepareCompletionExpr(
    uint64_t &completionPosition, StringRef exprText,
    MojoParserContext::REPLLocMapper::ExprLocMapper &locMapper,
    ArrayRef<std::pair<StringRef, Type>> variables, bool isFirstCell) {
  constexpr StringLiteral kCompletionMarker = "<#COMPLETION_MARKER#>";
  std::string exprTextWithMarker = exprText.substr(0, completionPosition).str();
  exprTextWithMarker += kCompletionMarker;
  exprTextWithMarker += exprText.drop_front(completionPosition).str();

  std::string wrappedExprText =
      wrapExpressionText(locMapper, "__mojo_repl_code_complete_fn",
                         exprTextWithMarker, variables, isFirstCell);

  completionPosition = wrappedExprText.find(kCompletionMarker);
  wrappedExprText.erase(completionPosition, kCompletionMarker.size());
  return wrappedExprText;
}

/// Maximum number of completion requests before forcing a full cache rebuild
/// to bound memory growth from stale completion modules. Configurable via the
/// MOJO_REPL_COMPLETION_CACHE_LIMIT environment variable.
static size_t getCompletionCacheRebuildThreshold() {
  static size_t threshold = [] {
    if (const char *env = std::getenv("MOJO_REPL_COMPLETION_CACHE_LIMIT"))
      if (int val = std::atoi(env); val > 0)
        return static_cast<size_t>(val);
    return size_t{50};
  }();
  return threshold;
}

/// Ensure the completion cache is up-to-date for a completion request.
/// Collects target modules, incrementally updates the cache, and returns the
/// buffer identifier for the completion buffer (for correct import resolution).
StringRef MojoParserContext::prepareCompletionCache(MojoASTDeclRef replDecl) {
  auto targetModules = collectCompletionTargetModules(
      impl->replModuleDecls, impl->prevReplModuleDecls, replDecl.decl);
  size_t target = targetModules.size();
  auto &mainSourceMgr = impl->sharedState.getSourceMgr();

  // Check if the existing cache can be incrementally extended.
  bool canExtend = impl->completionCache &&
                   impl->completionCache->replDeclKey == replDecl.decl &&
                   impl->completionCache->replayedModuleCount <= target &&
                   impl->completionCache->completionsSinceRebuild <
                       getCompletionCacheRebuildThreshold();

  if (!canExtend) {
    // Full rebuild: cache doesn't exist, replDecl changed, target set shrank,
    // or stale module threshold exceeded.
    impl->completionCache = std::make_unique<Impl::CompletionCache>(
        impl->sharedState.getContext(), impl->sharedState.options,
        mainSourceMgr);
    impl->completionCache->replDeclKey = replDecl.decl;
  }

  auto &cc = *impl->completionCache;
  for (size_t i = cc.replayedModuleCount; i < target; ++i) {
    cc.lastResolvedDecl = replayModuleIntoCache(
        *cc.context, cc.sourceMgr, mainSourceMgr, targetModules[i],
        MojoASTDeclRef(cc.lastResolvedDecl));
  }
  cc.replayedModuleCount = target;

  // Derive the buffer identifier from replDecl's source buffer so that
  // import resolution can find the correct working directory.
  if (replDecl) {
    const llvm::MemoryBuffer *origSourceBuf = getSourceMgr().getMemoryBuffer(
        getSourceMgr().FindBufferContainingLoc(replDecl->getLoc()));
    return origSourceBuf->getBufferIdentifier();
  }
  return "";
}

std::vector<Mojo::CodeCompletionResult>
MojoParserContext::codeCompleteREPLExpression(
    StringRef exprText, uint64_t completionPosition,
    ArrayRef<std::pair<StringRef, Type>> replVariables) {
  return codeCompleteREPLExpression(exprText, completionPosition, replVariables,
                                    MojoASTDeclRef());
}

std::vector<Mojo::CodeCompletionResult>
MojoParserContext::codeCompleteREPLExpression(
    StringRef exprText, uint64_t completionPosition,
    ArrayRef<std::pair<StringRef, Type>> replVariables,
    MojoASTDeclRef replDecl) {
  REPLLocMapper locMapper(getSourceMgr());
  locMapper.exprMappers.emplace_back(
      std::make_unique<REPLLocMapper::ExprLocMapper>(exprText));

  StringRef bufferIdentifier = prepareCompletionCache(replDecl);
  auto &cache = *impl->completionCache;

  std::string wrappedExprText = prepareCompletionExpr(
      completionPosition, exprText, *locMapper.exprMappers.back(),
      replVariables, !cache.lastResolvedDecl);

  // Run completion in the cached context. Each request adds a new module, but
  // lastResolvedDecl always points to the last *real* REPL module, so stale
  // completion modules are harmless.
  auto results = MojoParserContext::codeCompleteInContext(
      *cache.context, llvm::MemoryBufferRef(wrappedExprText, bufferIdentifier),
      completionPosition, [&](int fileId) {
        const auto *buf = cache.context->getSourceMgr().getMemoryBuffer(fileId);
        buildAndResolveREPLModule(buf, buf->getBufferIdentifier(),
                                  cache.context->getSharedState(),
                                  MojoASTDeclRef(cache.lastResolvedDecl),
                                  /*parseForLSP=*/false, replVariables);
        ++cache.completionsSinceRebuild;
      });

  // Filter out results pointing to internal decls.
  llvm::erase_if(results, [&](const Mojo::CodeCompletionResult &result) {
    return StringRef(result.label).starts_with("__mojo_repl");
  });
  return results;
}

std::optional<Mojo::SignatureHelpResult>
MojoParserContext::signatureHelpREPLExpression(
    StringRef exprText, uint64_t position,
    ArrayRef<std::pair<StringRef, Type>> replVariables,
    MojoASTDeclRef replDecl) {
  REPLLocMapper locMapper(getSourceMgr());
  locMapper.exprMappers.emplace_back(
      std::make_unique<REPLLocMapper::ExprLocMapper>(exprText));

  StringRef bufferIdentifier = prepareCompletionCache(replDecl);
  auto &cache = *impl->completionCache;

  std::string wrappedExprText =
      prepareCompletionExpr(position, exprText, *locMapper.exprMappers.back(),
                            replVariables, !cache.lastResolvedDecl);

  auto result = MojoParserContext::signatureHelpInContext(
      *cache.context, llvm::MemoryBufferRef(wrappedExprText, bufferIdentifier),
      position, [&](int fileId) {
        const auto *buf = cache.context->getSourceMgr().getMemoryBuffer(fileId);
        buildAndResolveREPLModule(buf, buf->getBufferIdentifier(),
                                  cache.context->getSharedState(),
                                  MojoASTDeclRef(cache.lastResolvedDecl),
                                  /*parseForLSP=*/false, replVariables);
        ++cache.completionsSinceRebuild;
      });

  return result;
}
