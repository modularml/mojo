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
// This file implements basic statement parsing.
//
//===----------------------------------------------------------------------===//

#include "ExprNodes.h"
#include "IREmitter.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/Constraints.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Mojo/MojoParser/Lexer.h"
#include "MojoUtils.h"
#include "OverloadSet.h"
#include "ParserBase.h"
#include "Support/Compiler/OperationUtils.h"

#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/LITDialect/LITAttrs.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/SpecialFunctions.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Mojo/ToolCommon/CompilationOptions.h"

#include "Mojo/MojoParser/ASTType.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/SourceMgr.h"
#include <filesystem>
#include <limits>

using namespace M::KGEN::LIT;
using namespace M::KGEN;
using namespace M;

//===----------------------------------------------------------------------===//
// Doc String support logic
//===----------------------------------------------------------------------===//

/// Parse the doc string and get its spelling into the specified string if
/// present, otherwise leave it empty.
void ParserBase::parseDocString(StringRef &docString) {
  // We don't want to treat things like "foo".print() as a doc string, so
  // consume any string toke and check to see if the next token is on a new
  // line.
  auto beforeStringCursor = LexerCursor(lexer);
  StringRef tokSpelling = getTokenSpelling();
  if (!consumeIf(Token::string))
    return; // Obviously no doc string.

  if (getToken().isStartOfLine())
    docString = tokSpelling;
  else // Start of another expression.
    beforeStringCursor.restore(lexer);
}

void ParserBase::parseDocString(ASTDecl &decl) {
  // The doc string is simply a follow-on string literal.
  StringRef docString;
  parseDocString(docString);
  if (docString.empty())
    return;
  if (auto astDeclOp =
          dyn_cast_or_null<ASTDeclInterface>(decl.getIfOperation())) {
    Location loc = shared.diags.translateLocation(
        lexer.getStringLiteralStartLoc(docString));

    astDeclOp.setDocStringAttr(DocStringAttr::get(
        StringAttr::get(getContext(), lexer.getStringLiteralValue(docString)),
        dyn_cast<FileLineColLoc>(loc)));
  }
}

//===----------------------------------------------------------------------===//
// Decorator support logic
//===----------------------------------------------------------------------===//

/// Return true if this token is the start of a statement that should not exist
/// on the same line as a @decorator specification. This is used to improve
/// error recovery.
static bool isStatementThatMightHaveDecorators(Token::Kind tokenKind) {
  switch (tokenKind) {
  case Token::kw_if:
  case Token::kw_for:
  case Token::kw_while:
  case Token::kw_try:
  case Token::kw_with:
  case Token::kw___match:
  case Token::kw_async:
  case Token::kw_def:
  case Token::kw_fn:
  case Token::kw_struct:
  case Token::kw_class:
  case Token::kw_from:
  case Token::kw_import:
  case Token::kw_pass:
  case Token::kw_var:
  case Token::kw_comptime:
  case Token::kw___mlir_region:
  case Token::kw_return:
  case Token::kw_raise:
  case Token::kw_continue:
  case Token::kw_break:
    return true;
  default:
    return false;
  }
}

SmallVector<std::pair<ExprNode *, LexerCursor>>
ParserBase::parseDecorators(ASTDecl &decl) {
  return parseDecorators(decl.getIndentation());
}

/// Parse any decorators that may be present for a statement at the specified
/// indentation level.  Note that this must be kept in sync with the logic in
/// parseStmt which skips over things until the right indentation level.
SmallVector<std::pair<ExprNode *, LexerCursor>>
ParserBase::parseDecorators(ssize_t indentation) {
  SmallVector<std::pair<ExprNode *, LexerCursor>> result;

  auto stopOnStatement = [&]() -> bool {
    return isStatementThatMightHaveDecorators(getToken().getKind());
  };

  llvm::SMLoc atLoc;
  while (consumeIf(Token::at, &atLoc)) {
    if (getToken().isStartOfLine()) {
      emitError(atLoc,
                "found stray '@'; '@' must be followed by a decorator name");
      skipUntilIndentation(indentation, /*stopOnSemicolon=*/false,
                           stopOnStatement);
      continue;
    }

    ExprNode *decoratorExpr;
    LexerCursor cursor = lexer.getCursor();
    if (parseExpression(decoratorExpr, indentation)) {
      skipUntilIndentation(indentation, /*stopOnSemicolon=*/false,
                           stopOnStatement);
      break;
    }
    result.push_back({decoratorExpr, cursor});

    if (!getToken().isStartOfLine() ||
        ssize_t(getToken().getIndentation().value()) > indentation) {
      emitTokenError("unexpected tokens after decorator, each need to be on "
                     "their own line");
      skipUntilIndentation(indentation, /*stopOnSemicolon=*/false,
                           stopOnStatement);
    }
  }
  // Decorators are applied to a decl starting from the one closest to it, so
  // reverse the vector.
  std::reverse(result.begin(), result.end());
  return result;
}

static void initializeBuilderForFunctions(ASTDecl &curDeclScope,
                                          OpBuilder &builder) {
  // Special logic for functions.
  if (auto funcOp = dyn_cast_or_null<FnOp>(curDeclScope.getIfOperation())) {
    // The default of inserting at the end of the function isn't right,
    // because it will have an lit.endfn: insert before that instead.
    assert(isa<EndFnOp>(funcOp.getBody()->back()) &&
           "expected lit.endfn at the end of a function body");
    builder.setInsertionPoint(&funcOp.getBody()->back());
  }
}

//===----------------------------------------------------------------------===//
// StmtParser
//===----------------------------------------------------------------------===//

/// This class either holds generated `LIT::LoopOp` or an error kind (not error
/// message) indicating where the error occurred, which helps to either keep
/// running parser to get more errors or stop parsing.
struct LoopResult {
  enum class ErrorKind {
    none,
    inLoopBody,
    inLoopStmt,
  };

  LoopResult(LIT::LoopOp loopOp) : loopOp(loopOp) {
    assert(loopOp && "expected non-nullptr `LIT:ForOp`");
  }

  LoopResult(ErrorKind kind) : error(kind) {
    assert(error != ErrorKind::none && "expected error kind to be set");
  }

  explicit operator bool() const { return loopOp != nullptr; }
  bool hasErrorInLoopBody() const {
    assert(!loopOp && "`LIT::ForOp` was successfully parsed");
    return error == ErrorKind::inLoopBody;
  }
  bool hasErrorInLoopStmt() const {
    assert(!loopOp && "`LIT::ForOp` was successfully parsed");
    return error == ErrorKind::inLoopStmt;
  }

  LIT::LoopOp getLoopOp() const {
    assert(loopOp && "expected non-nullptr `LIT:ForOp`");
    return loopOp;
  }

private:
  LIT::LoopOp loopOp = nullptr;
  ErrorKind error = ErrorKind::none;
};

/// This class provides the implementation details of the concrete Lightning
/// grammar.
namespace {
/// Decorators accepted on a `from ... import ...` statement.
struct FromImportDecorators {
  bool hasStableOverride = false;
  bool docInline = false;
};

struct StmtParser : public ParserBase {
  StmtParser(Lexer &lexer, ASTDecl &curDeclScope)
      : ParserBase(curDeclScope.getShared(), lexer), parentDecl(curDeclScope),
        curDeclScope(&curDeclScope), builder(curDeclScope.getDeclEndBuilder()) {
    initializeBuilderForFunctions(curDeclScope, builder);
  }
  StmtParser(Lexer &lexer, IREmitter &emitter)
      : ParserBase(emitter.shared, lexer), parentDecl(emitter.getDeclScope()),
        curDeclScope(&emitter.getDeclScope()),
        builder(emitter.builder ? emitter.builder.value()
                                : curDeclScope->getDeclEndBuilder()) {
    if (!emitter.builder)
      initializeBuilderForFunctions(emitter.getDeclScope(), builder);
  }

  ASTDecl &getDeclScope() const { return *curDeclScope; }

  ASTDecl &getParentDecl() { return parentDecl; }
  OpBuilder &getBuilder() { return builder; }

  /// Push a debug info lexical block to represent a local variable scope.
  void pushLocalScope(DebugInfo::DIBuilder::ScopeGuard &scopeGuard);

  // Expression emission.

  IREmitter getEmitter() { return IREmitter(*curDeclScope, builder); }

  /// Get an expression emitter for a parameter expression.
  IREmitter getParamEmitter(ExprContext context) {
    return IREmitter(*curDeclScope, context);
  }

  ParseResult parseSuite(ssize_t curIndent);
  void pushChildScope(DebugInfo::DIBuilder::ScopeGuard &scopeGuard,
                      llvm::SaveAndRestore<ASTDecl *> &keepDecl);
  /// Parse a "suite" body pushed under a local scope so any vardecls inside of
  /// it are popped at the end.
  ParseResult parseLocalScopeSuite(ssize_t curIndent);
  ParseResult parseStmt(bool onlySimpleStmt, bool &parsedCompound,
                        size_t stmtIndent);

  // Compound statements.
  ParseResult parseIfStmt(LexerCursor startCursor, size_t curIndent);
  ParseResult parseElif(Location ifLoc, LexerCursor startCursor,
                        size_t curIndent);
  ParseResult parseParamIf(Location ifLoc, LexerCursor startCursor,
                           size_t curIndent);
  ParseResult parseWhileStmt(size_t curIndent);
  ParseResult parseMatchStmt(size_t curIndent);

  // This emits the pattern for a 'for' loop, calling the specified 'bodyFn'
  // closure on success when in the scope of the loop, and the specified
  // 'errorFn' if there is a semantic error with the sequence expression or
  // target.
  // Return a struct with parsed ForOp or error kind to either abort parsing of
  // the parent suite or keep parsing it to get more errors.
  LoopResult emitForStmt(SMLoc forLoc, ExprNode *targetExpr, ExprNode *seqExpr,
                         std::function<LogicalResult()> bodyFn,
                         std::function<void()> errorFn = {});

  ParseResult parseForStmt(LexerCursor startCursor, size_t curIndent);
  ParseResult parseParamFor(size_t curIndent, SMLoc forLoc,
                            ExprNode *targetExpr, ExprNode *seqExpr);
  ParseResult parseTryStmt(size_t curIndent);
  ParseResult parseWithStmt(size_t curIndent);
  ParseResult parseSingleWithStmt(size_t curIndent, SMLoc smLoc, Location loc);
  ParseResult
  handleRaisingFinallyRegion(TryOp tryOp, SMLoc loc,
                             function_ref<ParseResult()> populateFinallyBody);

  // Comptime statements.
  // Handles 'comptime <keyword>' statements and 'comptime' variable decls.
  ParseResult parseComptimeCompoundStmt(LexerCursor startCursor,
                                        size_t curIndent, bool hadDecorators);
  // Parses 'comptime if' after 'comptime' has been consumed.
  ParseResult parseComptimeIfStmt(LexerCursor startCursor, size_t curIndent);
  // Parses 'comptime for' after 'comptime' has been consumed.
  ParseResult parseComptimeForStmt(LexerCursor startCursor, size_t curIndent);

  // Helper to parse 'for <target> in <seq>:' syntax.
  ParseResult parseForTargetAndSequence(size_t curIndent, SMLoc &forLoc,
                                        ExprNode *&targetExpr,
                                        ExprNode *&seqExpr);
  // Parses 'comptime assert' after keywords are consumed.
  ParseResult parseComptimeAssertStmtBody(LexerCursor startCursor,
                                          size_t curIndent, SMLoc kwLoc);
  ParseResult parseReturnStmt(size_t returnIndent);
  ParseResult parseRaiseStmt(size_t raiseIndent);
  ParseResult parseAssertStmt(size_t assertIndent);
  ParseResult parseBreakOrContinueStmt(Token::Kind kind, StringRef name,
                                       StringRef opName);

  // Declarations.
  ParseResult parseFromImportStmt(FromImportDecorators decorators);
  ParseResult parseImportStmt();
  /// Emit an error and return failure if the current scope is not a valid
  /// location for an import statement. Imports are permitted at module scope
  /// (FileModuleOp) and at function scope (FnOp), including inside comptime
  /// control-flow (ParamIfOp, ParamForOp) which is transparent to this check.
  /// Runtime control-flow bodies (HLCF::IfOp, LIT::LoopOp, etc.) are
  /// rejected. \p kwLoc should be the location of the leading `from` or
  /// `import` keyword so the diagnostic caret lands on the keyword.
  ParseResult checkImportScope(SMLoc kwLoc);
  ParseResult parseImportModuleName(SharedState::ImportPath &parsedName,
                                    bool allowRelativeImport);
  ParseResult parseDefFnStmt(LexerCursor startCursor, size_t curIndent);
  ParseResult parseStructStmt(LexerCursor startCursor, size_t curIndent);
  ParseResult parseTraitStmt(LexerCursor startCursor, size_t curIndent);
  ParseResult parseExtensionStmt(LexerCursor startCursor, size_t curIndent);
  ParseResult parseClassStmt(LexerCursor startCursor, size_t curIndent);
  ParseResult parseVarStmt(LexerCursor startCursor, size_t stmtIndent);
  // Parse an alias/comptime declaration. The keyword ('alias' or 'comptime')
  // must be consumed before calling this, and its location passed as kwLoc.
  ParseResult parseAliasDeclStmtBody(LexerCursor startCursor, size_t stmtIndent,
                                     SMLoc kwLoc);
  ParseResult parseMLIRRegionStmt(LexerCursor startCursor, size_t curIndent);

  // Helper invoked during parseDefFnStmt, meant to mark defaulted trait method
  // (first token in the function body is not '...').
  void maybeMarkDefaultedTraitMethod(FnOp fnOp);

  /// Check if we're inside a struct or trait body (where control flow
  /// statements are not allowed). Returns true if we are, false otherwise.
  bool isInTypeBody() const;

  /// Check if we're at module/file scope (where control flow statements are
  /// not allowed). Returns true if we are, false otherwise.
  bool isInModuleScope() const;

  /// Emit an error for a control flow statement that must be contained in a
  /// function. \p stmtName is the name of the statement (e.g., "if", "for",
  /// "comptime if"). \p loc is the location for the diagnostic.
  void emitControlFlowNotInFunctionError(SMLoc loc, StringRef stmtName);

private:
  /// This is parent declaration / scope that we're parsing into.
  ASTDecl &parentDecl;
  /// This is the current declaration / scope.
  ASTDecl *curDeclScope;

  /// This is the builder that we are constructing IR into.
  OpBuilder builder;
};
} // namespace

void StmtParser::pushLocalScope(DebugInfo::DIBuilder::ScopeGuard &scopeGuard) {
  SMLoc curLoc = getToken().getLoc();
  FileLineColLoc loc = shared.diags.translateLocation(curLoc);
  scopeGuard = shared.diBuilder->pushNestedLexicalBlock(
      shared.diBuilder->createFile(loc.getFilename()), loc.getLine(),
      loc.getColumn());
}

bool StmtParser::isInTypeBody() const {
  Operation *parent = parentDecl.getIfOperation();
  return parent && isa<StructDeclOp, TraitDeclOp, ExtensionDeclOp>(parent);
}

bool StmtParser::isInModuleScope() const {
  Operation *parent = parentDecl.getIfOperation();
  return isa_and_nonnull<LIT::FileModuleOp>(parent);
}

void StmtParser::emitControlFlowNotInFunctionError(SMLoc loc,
                                                   StringRef stmtName) {
  emitError(loc) << "'" << stmtName << "' must be contained in a function";
}

/// Parse a suite, which is either a series of comma separated simple_stmt's on
/// one line, or an indented block of statements. curIndent is the containing
/// statement's indentation level.
///
/// suite     ::=  [stmt_list NEWLINE] | NEWLINE INDENT statement+ DEDENT
/// statement ::=  stmt_list NEWLINE | compound_stmt
/// stmt_list ::=  simple_stmt (";" simple_stmt)* [";"]
ParseResult StmtParser::parseSuite(ssize_t curIndent) {
  // Ignore empty body at end of file: a `pass` is not required.
  if (getToken().is(Token::eof))
    return success();

  /// This function parses a stmt_list, and if simpleStmtOnly is false, it
  /// also allows a compound statement.
  auto parseStmtListOrCompound = [&](bool stmtListOnly,
                                     size_t stmtIndent) -> ParseResult {
    do {
      bool parsedCompound = false;
      if (parseStmt(/*onlySimpleStmt=*/stmtListOnly, parsedCompound,
                    stmtIndent))
        return failure();

      // If we parsed a compound statement, then we don't allow trailing
      // semicolons after it.
      if (parsedCompound)
        return success();

      // Otherwise, we parsed a simple statement, which means no more compound
      // statements are allowed.
      stmtListOnly = true;

      // Continue if we see a semicolon that isn't at the end of the line.
    } while (consumeIf(Token::semi) && !getToken().isStartOfLine());
    return success();
  };

  // If this suite is on the same line as the enclosing entity, just parse a
  // single stmt_list.
  if (!getToken().isStartOfLine())
    return parseStmtListOrCompound(
        /*stmtListOnly=*/true,
        /*stmtIndent=*/std::numeric_limits<size_t>::max());

  ssize_t indent = getToken().getIndentation().value();

  // If there is a newline, then parse a list of statements which can be either
  // a statement list or a compound_stmt.  Parse all the statements that are
  // more nested than this suite, and reject it if there are none.
  if (indent <= curIndent) {
    emitError(getTokenLocOrEndOfPreviousLineIfOnNewLine())
        << "body must not be empty; use 'pass' or check that the lines below "
           "are indented";
    return success();
  }

  // The first statement sets the expected indentation level for the whole body.
  auto bodyIndent = indent;
  SMLoc bodyIndentLoc = getToken().getLoc();
  while (getToken().isNot(Token::eof)) {
    if (!getToken().isStartOfLine())
      return emitTokenError("statements must start at the beginning of a line");

    indent = getToken().getIndentation().value();

    // If the indentation is less than we expect, then the suite is done.
    if (indent < bodyIndent)
      break;

    // Diagnose cases where the indentation is too great.
    if (indent > bodyIndent) {
      emitError(getToken().getLoc()) << "statement indentation must match the "
                                        "rest of the block; adjust to align";
    } else {
      bodyIndentLoc = getToken().getLoc();
    }

    if (parseStmtListOrCompound(/*stmtListOnly=*/false, indent))
      return failure();
  }
  return success();
}

void StmtParser::pushChildScope(DebugInfo::DIBuilder::ScopeGuard &scopeGuard,
                                llvm::SaveAndRestore<ASTDecl *> &keepDecl) {
  // If we are generating debug info, push a local scope
  if (shared.diBuilder)
    pushLocalScope(scopeGuard);

  // Push a new local variable scope.
  SMLoc loc;
  (void)getLocation(loc);
  curDeclScope = &getDeclResolver().addFullyResolvedDecl(nullptr, StringAttr(),
                                                         loc, curDeclScope);
}

/// Parse a "suite" body pushed under a local scope so any vardecls inside of
/// it are popped at the end.
ParseResult StmtParser::parseLocalScopeSuite(ssize_t curIndent) {
  DebugInfo::DIBuilder::ScopeGuard scopeGuard;
  llvm::SaveAndRestore<ASTDecl *> keepDecl(curDeclScope);
  // Push a new local variable scope for the subsequent suite.
  pushChildScope(scopeGuard, keepDecl);

  // Forward to the normal suite parse method.
  return parseSuite(curIndent);
}

/// Emit a warning when an expression is emitted at statement context, and it
/// returns an unused result.
static void diagnoseIgnoredResult(const ExprNode *expr, CValue value,
                                  SharedState &shared) {
  ASTType valueType = value.getRValueType();

  // Return true if the specified type can be implicitly ignored.
  // TODO(MOCO-32):
  //  Should have a better way to say that it is safe to
  //  implicitly ignore a value of a type (e.g. a type decorator)
  auto isImplicitlyIgnorableType = [&](ASTType type) -> bool {
    if (type.isNoneType() || sugarIsa<NeverType, TypeCheckErrorType>(type))
      return true;

    // Allow object/PythonObject to be ignored.  This should really be
    // implemented with a decorator on the type, not hard coded here.
    auto declRef = sugarDynCast<LIT::StructType>(type.mlirType);
    if (!declRef || !declRef.getParamValues().empty())
      return false;

    // object is implicitly returned by 'def's, PythonObject is pervasive in
    // interop.
    StringRef name = declRef.getName().getValue();
    return name == "object" || name == "PythonObject" || name == "NoneType";
  };

  if (isImplicitlyIgnorableType(valueType) ||
      // The `x = y` operation returns a borrowed version of its operand but its
      // result can be ignored.  "var x: T" patterns can also be ignored.
      expr->kind == ExprNode::kAssign || expr->kind == ExprNode::kTypePattern)
    return;

  // If this type is a function with no formal arguments and an ignorable type,
  // we emit a warning with a fix it hint suggesting that it get called.
  // TODO: This is incorrect for default arguments and varargs.
  if (auto sig = FnOrFnLiteralTypeGeneratorType::tryGet(valueType)) {
    // Get the result type without any error handling in the way.
    Type resultType = sig->getUserResultType();
    if ((sig->getNumArguments() ==
         ((unsigned)sig->hasMemoryOnlyResult() + (unsigned)sig->isThrows())) &&
        isImplicitlyIgnorableType(resultType)) {
      shared.emitWarning(expr->getLoc())
          << "function is not called; add '()' after name" << expr->getRange()
          << FixIt::insertAfterToken(expr->getRange().getEnd(), "()",
                                     shared.diags);
      return;
    }
  }

  // If the expression returned an unawaited value, then the expression should
  // be awaited. Check for an '__await__' function.
  // TODO: This should be handled with linear types.
  if (shared.typeHasMember(valueType, "__await__", expr->getLoc())) {
    shared.emitWarning(expr->getLoc())
        << valueType << " value is not awaited; use 'await' to get its result"
        << expr->getRange()
        << FixIt::insertBeforeToken(expr->getRangeStart(), "await ");
    return;
  }

  // Otherwise emit a warning, and suggest assigning to _.
  auto startLoc = expr->getRange().getStart();
  shared.emitWarning(expr->getLoc())
      << valueType << " value is unused; assign to '_' to discard the result"
      << expr->getRange() << FixIt::insertBeforeToken(startLoc, "_ = ");
}

/// When `onlySimpleStmt` is true, this parses the simple_stmt production,
/// otherwise it parses the broader `statement` production that includes
/// compound statements.  This sets `parsedCompound` to true if
/// `onlySimpleStmt` was false and we parsed a compound stmt.
///
/// statement ::= compound_stmt | simple_stmt
///
/// compound_stmt ::= if_stmt
///                 | while_stmt
///                 | for_stmt
///                 | try_stmt
///                 | with_stmt
///                 | match_stmt [TODO]
///                 | funcdef
///                 | structdef
///                 | classdef [TODO]
///                 | async_with_stmt [TODO]
///                 | async_for_stmt [TODO]
///                 | async_funcdef [TODO]
///
/// simple_stmt ::= expression_stmt
///               | assert_stmt [TODO]
///               | alias_decl_stmt
///               | var_decl_stmt
///               | assignment_stmt
///               | augmented_assignment_stmt
///               | annotated_assignment_stmt [TODO]
///               | pass_stmt
///               | del_stmt [TODO]
///               | return_stmt
///               | disable_del_stmt
///               | yield_stmt [TODO]
///               | raise_stmt [TODO]
///               | break_stmt [TODO]
///               | comptime_stmt
///               | continue_stmt [TODO]
///               | import_stmt
///               | future_stmt [TODO]
///               | global_stmt [TODO]
///               | nonlocal_stmtParseResult [TODO]
///
// Forward declaration — defined below after parseFromImportStmt.
static FromImportDecorators parseFromImportDecorators(SharedState &shared,
                                                      LexerCursor startCursor,
                                                      size_t stmtIndent);

ParseResult StmtParser::parseStmt(bool onlySimpleStmt, bool &parsedCompound,
                                  size_t stmtIndent) {
  // Generate pretty stack traces if a crash happens in this scope.
  CrashReporter crashReporter(getToken().getLoc(), "parsing statement", shared);

  // This is the cursor for the start of the declaration, that will be used in
  // the signature resolution phase.
  LexerCursor startCursor = getLexer().getCursor();

  // This lambda is used to generate an error when a compound statement is used
  // in a scenario that expects simple statements.
  auto rejectSimpleStmt = [&]() {
    parsedCompound = true; // Tell the caller that we parsed a compound stmt.
    if (!onlySimpleStmt)
      return;
    emitTokenError() << "'" << getToken().getSpelling()
                     << "' statement must be on its own line";
  };

  // Check if we're in a non-function scope (type body or module scope) where
  // control flow statements are not allowed. Emits an error and recovers.
  auto rejectInNonFunctionScope = [&]() -> bool {
    if (!isInTypeBody() && !isInModuleScope())
      return false;
    emitControlFlowNotInFunctionError(getToken().getLoc(),
                                      getToken().getSpelling());
    // Consume the keyword token first so skipUntilIndentation makes progress.
    consumeToken();
    skipUntilIndentation(stmtIndent);
    return true;
  };

  // Skip over any decorators that are present.  These will be reparsed during
  // signature resolution phase of a declaration.
  while (consumeIf(Token::at)) {
    auto stopOnStatement = [&]() -> bool {
      return isStatementThatMightHaveDecorators(getToken().getKind());
    };

    skipUntilIndentation(stmtIndent, /*stopOnSemicolon=*/false,
                         stopOnStatement);

    // If the next token isn't indented, but is the start of a statement, then
    // these decorators are incorrectly on the same line as the statement.
    // Reject with a specific error message and ignore the whole thing.
    if (!getToken().isStartOfLine() && stopOnStatement()) {
      emitError(startCursor.getToken().getLoc())
          << "decorators must be on their own line; add a newline after the "
             "decorator";
      // Skip the body of the statement entirely.
      skipUntilIndentation(stmtIndent);
      return success();
    }

    // If the next token is for a less indented declaration, then this is a
    // floating decorator not necessarily attached to it.  Ignore the
    // decorators and let the outer level of the parser keep finding stuff.
    // This leads to better error recovery.
    if (getToken().isStartOfLine() &&
        getToken().getIndentation().value() < stmtIndent) {
      emitError(startCursor.getToken().getLoc())
          << "decorator must be followed by a definition on the next line; "
             "remove any blank lines between them";
      return success();
    }
  }

  // This emits an error message if we parsed a decorator, because this
  // statement doesn't support them.
  bool hadDecorators = (startCursor != getLexer().getCursor());
  auto rejectDecorator = [&](bool inFunctionBody = false,
                             bool printToken = true) {
    if (!hadDecorators)
      return;
    auto diag = emitTokenError();
    if (printToken)
      diag << "'" << getToken().getSpelling() << "' ";
    diag << "statement " << (inFunctionBody ? "in function body " : "")
         << "does not support decorators; remove the decorator";
  };

  switch (getToken().getKind()) {
    //===------------------------------------------------------------------===//
    // Compound statements.
    //===------------------------------------------------------------------===//
  case Token::kw_if:
    rejectDecorator();  // Decorators not allowed.
    rejectSimpleStmt(); // Not a simple_stmt.
    if (rejectInNonFunctionScope())
      return success();
    return parseIfStmt(startCursor, stmtIndent);
  case Token::kw_for:
    rejectDecorator();  // Decorators not allowed.
    rejectSimpleStmt(); // Not a simple_stmt.
    if (rejectInNonFunctionScope())
      return success();
    return parseForStmt(startCursor, stmtIndent);
  case Token::kw_while:
    rejectDecorator();  // Decorators not allowed.
    rejectSimpleStmt(); // Not a simple_stmt.
    if (rejectInNonFunctionScope())
      return success();
    return parseWhileStmt(stmtIndent);
  case Token::kw___match:
    rejectDecorator();  // Decorators not allowed.
    rejectSimpleStmt(); // Not a simple_stmt.
    if (rejectInNonFunctionScope())
      return success();
    return parseMatchStmt(stmtIndent);
  case Token::kw_try:
    rejectDecorator(); // Decorators not allowed.
    rejectSimpleStmt();
    if (rejectInNonFunctionScope())
      return success();
    return parseTryStmt(stmtIndent);
  case Token::kw_with:
    rejectDecorator(); // Decorators not allowed.
    rejectSimpleStmt();
    if (rejectInNonFunctionScope())
      return success();
    return parseWithStmt(stmtIndent);
  case Token::kw_async:
  case Token::kw_def:
  case Token::kw_fn:
    rejectSimpleStmt(); // Not a simple_stmt.
    return parseDefFnStmt(startCursor, stmtIndent);
  case Token::kw_struct:
    rejectSimpleStmt(); // Not a simple_stmt.
    return parseStructStmt(startCursor, stmtIndent);
  case Token::kw_trait:
    rejectSimpleStmt(); // Not a simple_stmt.
    return parseTraitStmt(startCursor, stmtIndent);
  case Token::kw___extension:
    rejectSimpleStmt(); // Not a simple_stmt.
    return parseExtensionStmt(startCursor, stmtIndent);
  case Token::kw_class:
    rejectSimpleStmt(); // Not a simple_stmt.
    return parseClassStmt(startCursor, stmtIndent);

    //===------------------------------------------------------------------===//
    // Simple statements.
    //===------------------------------------------------------------------===//
  case Token::kw_from: {
    FromImportDecorators decorators =
        hadDecorators
            ? parseFromImportDecorators(shared, startCursor, stmtIndent)
            : FromImportDecorators{};
    return parseFromImportStmt(decorators);
  }
  case Token::kw_import:
    rejectDecorator(); // Decorators not allowed.
    return parseImportStmt();
  case Token::kw_pass:
  case Token::dot_dot_dot:
    // pass_stmt ::= "pass"
    consumeToken();
    return success();
  case Token::kw_var:
    // In function bodies, we parse 'var' as expressions, part of the pattern
    // grammar.  TODO: extend this to other contexts.
    if (isa_and_nonnull<FnOp>(getParentDecl().getIfOperation()))
      break;
    return parseVarStmt(startCursor, stmtIndent);
  case Token::kw_comptime:
    return parseComptimeCompoundStmt(startCursor, stmtIndent, hadDecorators);
  case Token::kw___mlir_region:
    rejectDecorator();
    rejectSimpleStmt();
    return parseMLIRRegionStmt(startCursor, stmtIndent);
  case Token::kw_return:
    rejectDecorator(); // Decorators not allowed.
    return parseReturnStmt(stmtIndent);
  case Token::kw_raise:
    rejectDecorator(); // Decorators not allowed.
    return parseRaiseStmt(stmtIndent);
  case Token::kw_assert:
    rejectDecorator(); // Decorators not allowed.
    return parseAssertStmt(stmtIndent);
  case Token::kw_continue:
    rejectDecorator(); // Decorators not allowed.
    return parseBreakOrContinueStmt(Token::kw_continue, "continue",
                                    LIT::ContinueOp::getOperationName());
  case Token::kw_break:
    rejectDecorator(); // Decorators not allowed.
    return parseBreakOrContinueStmt(Token::kw_break, "break",
                                    LIT::BreakOp::getOperationName());
  default:
    break;
  }

  // Parse a single expression, an assignment stmt, or augmented assignment
  // statement.
  ExprNode *expr = nullptr;
  bool isVarStatement = getToken().is(Token::kw_var);
  rejectDecorator(/*inFunctionBody=*/isVarStatement,
                  /*printToken=*/isVarStatement);
  if (parseSimpleStmtExprs(expr, stmtIndent))
    return failure();

  // We cannot emit general statements unless we're in a function body.
  if (!isa_and_nonnull<FnOp, UnboundRegionOp>(
          getParentDecl().getIfOperation())) {
    // TODO: Top level expressions will be supported in the future.
    if (isa_and_nonnull<FileModuleOp>(parentDecl.getIfOperation())) {
      emitError(startCursor.getToken().getLoc())
          << "expressions must not appear at file scope; move this into a "
             "function body";
    } else if (isa_and_nonnull<StructDeclOp>(parentDecl.getIfOperation())) {
      emitError(startCursor.getToken().getLoc())
          << "bare expressions must not appear within structs; use 'var' for a "
             "field or move this into a method body";
    } else if (isa_and_nonnull<TraitDeclOp>(parentDecl.getIfOperation())) {
      emitError(startCursor.getToken().getLoc())
          << "bare expressions must not appear within traits; use 'comptime' "
             "for an associated type or move this into a method body";
    } else if (isa_and_nonnull<ExtensionDeclOp>(parentDecl.getIfOperation())) {
      emitError(startCursor.getToken().getLoc())
          << "bare expressions must not appear within extensions; move this "
             "into a method body";
    } else {
      // fallback: unknown parent declaration type
      emitError(expr->getLoc(),
                "bare expression must not appear here; move this into a "
                "function body")
          << expr->getRange();
    }
    return success();
  }

  // Emit the expression and ignore the results.  If it is an assignment
  // statement, it will return None.  Other expressions can return whatever they
  // will naturally return.
  auto emitter = getEmitter();
  // Result is ignored, so we don't care where it goes.
  CValue result = emitter.emitExprCValue(expr, EC_TopLevelStmt);
  if (!result)
    return success();

  // Emit a warning if the result is a value we should warn when unused.
  diagnoseIgnoredResult(expr, result, shared);
  return success();
}

//===----------------------------------------------------------------------===//
// Simple statements.
//===----------------------------------------------------------------------===//

/// Handles statements starting with the 'comptime' keyword.
///
/// 'comptime' can introduce either:
///   1. Compound statements like 'comptime assert expr [, msg]'
///   2. Variable declarations like 'comptime x = 4'
///
/// This function consumes 'comptime', checks what follows, and dispatches
/// accordingly. It is extensible for future 'comptime <keyword>' statements.
///
/// The hadDecorators parameter indicates whether decorators were parsed before
/// the 'comptime' keyword (used for decorator rejection since we can't use the
/// standard rejectDecorator lambda after consuming tokens).
ParseResult StmtParser::parseComptimeCompoundStmt(LexerCursor startCursor,
                                                  size_t curIndent,
                                                  bool hadDecorators) {
  SMLoc kwLoc = consumeToken(Token::kw_comptime).getLoc();

  auto rejectDecorator = [&](bool inFunctionBody = false,
                             bool isCompoundStmt = true) {
    if (!hadDecorators)
      return;

    auto diag = emitTokenError();
    diag << "'comptime";
    if (isCompoundStmt)
      diag << " " << getToken().getSpelling();
    diag << "' statement " << (inFunctionBody ? "in function body " : "")
         << "does not support decorators; remove the decorator";
  };

  // Check if we're in a non-function scope (type body or module scope) where
  // comptime control flow statements are not allowed. Emits an error and
  // recovers.
  auto rejectInNonFunctionScope = [&](StringRef stmtKind) -> bool {
    if (!isInTypeBody() && !isInModuleScope())
      return false;
    std::string stmtName = ("comptime " + stmtKind).str();
    emitControlFlowNotInFunctionError(kwLoc, stmtName);
    // Consume the keyword token first so skipUntilIndentation makes progress.
    consumeToken();
    skipUntilIndentation(curIndent);
    return true;
  };

  // Dispatch based on the current keyword (after 'comptime').
  switch (getToken().getKind()) {
  case Token::kw_assert:
    rejectDecorator();
    consumeToken(Token::kw_assert);
    return parseComptimeAssertStmtBody(startCursor, curIndent, kwLoc);
  case Token::kw_if:
    rejectDecorator();
    if (rejectInNonFunctionScope("if"))
      return success();
    return parseComptimeIfStmt(startCursor, curIndent);
  case Token::kw_for:
    rejectDecorator();
    if (rejectInNonFunctionScope("for"))
      return success();
    return parseComptimeForStmt(startCursor, curIndent);
  default:
    break;
  }

  // Reject statement and declaration keywords with a clear
  // "cannot be used with" error.
  // Other keywords like 'True' or 'or' pass through to the alias parser
  // which reports "identifier expected" (likely typos, not semantic errors).
  if (getToken().isStatementKeyword() || getToken().isDeclKeyword()) {
    return emitTokenError() << "'comptime' cannot be used with '"
                            << getToken().getSpelling() << "'",
           failure();
  }

  // This is a comptime variable declaration (e.g., 'comptime x = 4').
  // Decorators on aliases are not allowed inside function bodies.
  if (isa_and_present<FnOp>(getParentDecl().getIfOperation()))
    rejectDecorator(/*inFunctionBody=*/true, /*isCompoundStmt=*/false);
  return parseAliasDeclStmtBody(startCursor, curIndent, kwLoc);
}

/// if_stmt ::=  "if" assignment_expression ":" suite
///             ("elif" assignment_expression ":" suite)*
///             ["else" ":" suite]
ParseResult StmtParser::parseIfStmt(LexerCursor startCursor, size_t curIndent) {
  Location ifLoc = translateLocation(getToken().getLoc());
  if (parseToken(Token::kw_if, "expected 'if' token after decorators"))
    return failure();
  return parseElif(ifLoc, startCursor, curIndent);
}

/// Parses 'comptime if <condition>:' after 'comptime' has been consumed.
/// Delegates to parseParamIf for the actual IR generation.
ParseResult StmtParser::parseComptimeIfStmt(LexerCursor startCursor,
                                            size_t curIndent) {
  Location ifLoc = translateLocation(getToken().getLoc());
  consumeToken(Token::kw_if);
  return parseParamIf(ifLoc, startCursor, curIndent);
}

/// Helper to parse 'for <target> in <seq>:' syntax, used by both regular
/// and comptime for statements. Returns the parsed target and sequence exprs.
ParseResult StmtParser::parseForTargetAndSequence(size_t curIndent,
                                                  SMLoc &forLoc,
                                                  ExprNode *&targetExpr,
                                                  ExprNode *&seqExpr) {
  forLoc = consumeToken(Token::kw_for).getLoc();

  // parse [target_list] in [starred_list]
  // for now, we expect target_list to be an identifier
  // the [starred_list] needs to be a sequence with a __iter__ method.
  if (parseTargetListExpr(targetExpr, curIndent) ||
      parseToken(Token::kw_in, "expected 'in' after target identifier"))
    return failure();

  if (parseExpression(seqExpr) ||
      parseToken(Token::colon, "expected ':' after expression"))
    return failure();

  return success();
}

/// Parses 'comptime for <target> in <seq>:' after 'comptime' has been consumed.
/// Delegates to parseParamFor for the actual IR generation.
ParseResult StmtParser::parseComptimeForStmt(LexerCursor startCursor,
                                             size_t curIndent) {
  SMLoc forLoc;
  ExprNode *targetExpr = nullptr;
  ExprNode *seqExpr = nullptr;
  if (parseForTargetAndSequence(curIndent, forLoc, targetExpr, seqExpr))
    return failure();

  llvm::SaveAndRestore builderSaver(builder);
  return parseParamFor(curIndent, forLoc, targetExpr, seqExpr);
}

/// Parses a comptime assert statement after the keywords have been consumed.
///
/// Grammar:
///   comptime_assert_stmt ::= "comptime" "assert" expression ["," message]
///                          | "__comptime_assert" expression ["," message]
///
/// The caller must consume the keyword(s) before calling this function:
///   - For 'comptime assert': both 'comptime' and 'assert' are consumed
///   - For '__comptime_assert': the single keyword is consumed
///
/// kwLoc is the location of the first keyword, used for error reporting.
ParseResult StmtParser::parseComptimeAssertStmtBody(LexerCursor startCursor,
                                                    size_t curIndent,
                                                    SMLoc kwLoc) {
  ExprNode *expr = nullptr;
  if (parseExpression(expr, curIndent))
    return failure();

  ExprNode *messageExpr = nullptr;
  if (consumeIf(Token::comma))
    if (parseExpression(messageExpr, curIndent))
      return failure();

  // Ok, now that we parsed all the tokens for this statement, do semantic
  // analysis. First ensure we're in a function. This may be relaxed later.
  if (!isa_and_nonnull<FnOp>(getParentDecl().getIfOperation())) {
    emitError(kwLoc, "'comptime assert' must be inside a function; move this "
                     "into a function body");
    return success();
  }

  IREmitter emitter = getParamEmitter(EC_ComptimeAssert);
  RValue prop = emitter.emitExprScalarBool(expr, EC_ComptimeAssert);
  if (!prop)
    return failure();
  PValue propVal = prop.getIfPValue();
  if (!propVal) {
    emitter.emitErrorForDynamicValueInParameter(expr->getLoc());
    return failure();
  }

  TypedAttr message;
  if (messageExpr) {
    IREmitter paramEmitter = emitter.getParamEmitter(EC_ComptimeAssert);
    CValue messageVal =
        paramEmitter.emitExprCValue(messageExpr, EC_ComptimeAssert);
    message = paramEmitter.emitStringExprAsDataToStr(messageVal, messageExpr,
                                                     kwLoc, EC_ComptimeAssert);
    if (!message)
      return failure();
  } else {
    message = StringAttr::get({}, KGEN::StringType::get(builder.getContext()));
  }

  // If the condition is always True, the assertion is redundant and adds no
  // useful constraint, so skip emitting IR entirely.
  if (auto simdAttr = sugarDynCast<SIMDAttr>(propVal.get());
      simdAttr && simdAttr.getAsBool())
    return success();

  Location loc = translateLocation(kwLoc);
  auto assertOp =
      KGEN::ParamAssertOp::create(builder, loc, propVal.get(), message);

  // All subsequent statements implicitly belong to a new, nested parameter
  // scope. This scope completely takes over as the new `curDeclScope` and the
  // old scope is not restored.
  curDeclScope = &getDeclResolver().addFullyResolvedDecl(assertOp, StringAttr(),
                                                         kwLoc, curDeclScope);
  // Inject this assumption into the newly created context.
  // Convert `x and y` to `x & y` so we get better canonicalization.
  TypedAttr deShortCircuitCond = LIT::deShortCircuitCond(propVal.get());
  // This constraint is an assumption (a premise the scope now knows to hold),
  // not a checked constraint, so it carries no failure message: an assumption
  // is never the subject of a violation diagnostic. The assert's own message
  // lives on the `ParamAssertOp` above and surfaces if the assert fails.
  curDeclScope->insertKnownAssumptions(
      {ConstraintAttr::get(deShortCircuitCond, loc, /*message=*/StringAttr())});
  return success();
}

/// return_stmt ::= "return" [expression_list]
ParseResult StmtParser::parseReturnStmt(size_t returnIndent) {
  auto loc = consumeToken(Token::kw_return).getLoc();

  // If there is an expression list present, parse it.
  ExprNode *operandExpr = nullptr;
  if (isTokenInCurrentStatement(returnIndent)) {
    if (parseExpressionList(operandExpr, returnIndent))
      return failure();
  }

  // Ok, now that we parsed all the tokens for this statement, do semantic
  // analysis.  First ensure we're in a function.
  auto func = dyn_cast_or_null<FnOp>(getParentDecl().getIfOperation());
  if (!func) {
    emitError(
        loc,
        "'return' must be inside a function; move this into a function body");
    return success();
  }

  auto emitter = getEmitter();
  FnTypeGeneratorType declSig = func.getFuncTypeGenerator();

  // Next check the forms: We may or may not have a result expression.  If the
  // result expression is missing and we have a named result slot, just emit a
  // normal return with no value, assuming it has already been assigned.
  if (!operandExpr && func.getNamedResultAttr()) {
    emitter.emitNormalReturn(translateLocation(loc), /*resultVal*/ {},
                             /*emitEndFunc=*/false);
    return success();
  }

  // Otherwise, we treat a missing result as "None", which can implicitly
  // convert to whatever the expected type is.  This also keeps the logic below
  // unified.
  SimpleLiteralNode noneExpr(ExprNode::kNoneLiteral, loc);
  if (!operandExpr)
    operandExpr = &noneExpr;

  // Materialize the expression values into IR.
  AnyValue resultValue;

  // Figure out where to emit the value. If the result is memory-only, return
  // into the result slot, otherwise just ensure the right SRValue type.
  ASTType userResultType = func.getUserResultType();
  ExprDest resultDest(userResultType, EC_ReturnValue);

  if (declSig.hasMemoryOnlyResult())
    resultDest = ExprDest(MLValue(func.getArguments().back()), EC_ReturnValue);

  if (!declSig.isRefResult()) {
    // Convert the returned value to the returned type of the function.
    resultValue = emitter.emitExpr(operandExpr, resultDest);
    if (!resultValue) {
      resultDest.resetForError(emitter);
      return {};
    }
  } else {
    // When returning a reference, emit it as an MValue then coerce.
    auto resultCValue = emitter.emitExprCValue(
        operandExpr, EC_ReturnValue, userResultType.getReferenceElementType());
    if (!resultCValue) {
      resultDest.resetForError(emitter);
      return {};
    }

    Value refValue =
        emitter.emitRefValue({resultCValue, operandExpr}, EC_ReturnValue);
    if (!refValue) {
      resultDest.resetForError(emitter);
      return {};
    }
    RefType argType = sugarCast<RefType>(refValue.getType());

    // We already checked the element type, check the origin and address
    // space.
    // TODO: Move this to general implicit conversion diagnostics.
    if (!userResultType.isEqualCanon(argType)) {
      if (!emitter.canZeroCostConvert(argType, userResultType).isTrue()) {
        auto expectedRefType = sugarCast<RefType>(userResultType);
        auto diag = emitter.emitError(operandExpr->getLoc())
                    << "cannot return reference with incompatible ";
        if (argType.getOrigin() != expectedRefType.getOrigin()) {
          // See if the origins agree with mutability stripped.  If not,
          // complain about the base to avoid the mutcast in the diagnostic.
          auto argO = OriginMutCastAttr::strip(argType.getOrigin());
          auto expO = OriginMutCastAttr::strip(expectedRefType.getOrigin());
          if (argO != expO) {
            diag << "origin: '" << ASTType::getOriginAsString(argO, &shared)
                 << "' vs '" << ASTType::getOriginAsString(expO, &shared)
                 << "'";
          } else {
            diag << "origin mutability: " << argType.isMutable() << " vs "
                 << expectedRefType.isMutable();
          }
        } else {
          assert(argType.getAddressSpace() !=
                     expectedRefType.getAddressSpace() &&
                 "Only origin and address space can disagree given the "
                 "element types agree");
          diag << "address space: " << argType.getAddressSpace() << " vs "
               << expectedRefType.getAddressSpace();
        }
        resultDest.resetForError(emitter);
        return {};
      }
    }

    // We're returning the reference itself, so switch to SRValue and emit to
    // the ExprDest. This does implicit conversions to the expected type (e.g.
    // to a more general origin).
    resultValue =
        emitter.emitRValue({SRValue(refValue), operandExpr}, resultDest);
    if (!resultValue) {
      resultDest.resetForError(emitter);
      return success();
    }
  }

  // Check for a common error of explicitly returning the name from a named
  // result function.  This will turn into an exclusivity error downstream
  // (reading and writing to the same result slot), so we want to reject it in
  // the parser for better QoI.  Otherwise we get "use of uninit value".
  //
  // We do this syntactically because we have to emit the input expression
  // directly into the result slot (e.g. we could be calling a function that
  // returns a non-copyable type) and doing so with this pattern will create a
  // copy+move into and out of a temporary.
  if (auto resultName = func.getNamedResultAttr()) {
    if (auto dre = dyn_cast<DeclRefNode>(operandExpr->getWithoutParens()))
      if (dre->spelling == resultName.strref()) {
        auto diag = emitter.emitError(loc);
        diag << "'out' argument cannot be returned by name; remove the return "
                "statement entirely or change it to just 'return'";
        diag.attachNote(operandExpr->getLoc())
            << "remove the expression if the return slot is already "
               "initialized"
            << FixIt::remove(operandExpr->getRange());
        return success();
      }
  }

  // If the result is supposed to be an SRValue, ensure we promote to a
  // register we can return.
  Value resultVal;
  if (!declSig.hasMemoryOnlyResult()) {
    resultVal = emitter.emitSRValue({resultValue, operandExpr}, EC_ReturnValue,
                                    func.getMLIRResultType());
    if (!resultVal)
      return {};
  }

  // Finally emit the normal result value, handling things like throwing results
  // and None logic.
  emitter.emitNormalReturn(translateLocation(loc), resultVal,
                           /*emitEndFunc=*/false);
  return success();
}

/// Given an insertion point in a block, scan up the parent hierarchy to see if
/// this block is nested under a try.  If so, return that operation and whether
/// this block is nested within the 'except' part of the operation.  If
/// we are currently in the 'else' part of a try, we keep scanning since the try
/// isn't relevant.
static std::pair<TryOp, bool> findParentTry(Block *currentBlock) {
  while (Operation *parentOp = currentBlock->getParentOp()) {
    // If we hit the top of the function we aren't nested.
    if (isa<FnOp>(parentOp))
      break;

    // If this is a try, determine which region we're in.

    TryOp tryOp = dyn_cast<TryOp>(parentOp);
    if (tryOp) {
      if (&tryOp.getTryRegion().front() == currentBlock)
        return {tryOp, false};
      if (&tryOp.getExceptRegion().front() == currentBlock)
        return {tryOp, true};

      // Must be in the else, which doesn't stop propagation.
      assert(&tryOp.getElseRegion().front() == currentBlock);
    }

    // If this is not a try op, keep scanning.
    currentBlock = parentOp->getBlock();
  }

  // Didn't find a try.
  return {TryOp(), false};
}

/// Inject a call to a special method that the debugger stops at when
/// supporting exception/error breakpoints.
static LogicalResult injectDebuggerRaiseHookCall(SharedState &shared,
                                                 IREmitter &emitter,
                                                 ASTDecl &declContext,
                                                 llvm::SMLoc loc,
                                                 const ExprNode *node) {
  ArrayRef<ASTDecl *> raiseHookFns =
      shared.getBuiltinFunction(declContext, {"std", "builtin", "error"},
                                "__mojo_debugger_raise_hook", loc);
  if (raiseHookFns.empty())
    return failure();

  ParamBindings bindings(declContext, node);
  OverloadSet call("__mojo_debugger_raise_hook", raiseHookFns,
                   std::move(bindings), CallSyntax::kDirectCall);
  call.emitCall(CallOperands{CallSyntax::kDirectCall, node, EC_RaiseValue, {}},
                emitter);

  // Emit a LineTableLocOp after the call to ensure the instruction after the
  // call has the correct debug location. This is needed for ARM where StepOut()
  // lands on the return address (the instruction after the call), and without
  // this the debugger would report the wrong line number for inlined functions.
  if (emitter.builder && shared.diBuilder) {
    Location scopedLoc =
        shared.diBuilder->createScopedLoc(shared.diags.translateLocation(loc));
    DebugInfo::LineTableLocOp::create(*emitter.builder, scopedLoc);
  }

  return success();
}

ParseResult StmtParser::parseRaiseStmt(size_t raiseIndent) {
  llvm::SMRange loc = consumeToken(Token::kw_raise).getLocRange();

  ExprNode *errorExpr = nullptr;
  // If there is an error expression, parse it.
  if (isTokenInCurrentStatement(raiseIndent) &&
      parseExpression(errorExpr, raiseIndent))
    return failure();

  // TODO: Support "from" exception chaining.

  // Ok, we are syntactically sound.  Check to see if we're in a try block, and
  // (if so) whether we are in.  Python's notion of a current exception is fully
  // dynamic, which we don't support yet.  For now, we only support 'raise' with
  // no expression in the 'except' block of a 'try'.
  //
  //    def foo(): raise   # Rethrow any currently-being-handled exception
  //    try:
  //      print(1/0)
  //    except Exception as exc:
  //      print("hello")
  //      foo()   # rethrows the caught exception
  //

  // Find the nearest error slot if the parser is in a context that can raise.
  auto emitter = getEmitter();
  MLValue errSlot = emitter.findNearestErrorSlot();
  if (!errSlot) {
    emitError(loc.Start, "'raise' requires a surrounding 'try' block or the "
                         "enclosing function to declare 'raises'")
        << loc;
    return success();
  }

  ExprDest dest(errSlot, EC_RaiseValue);
  // If the contextual caught type is unresolved, then we're the first raise
  // in a try block.  Resolve the error type to whatever type we are raising.
  bool inferringErrorType = isa<UnresolvedType>(errSlot.getRValueType());
  if (inferringErrorType) {
    auto errorVar = cast<VarDeclOp>(errSlot.getDefiningOp());
    dest = ExprDest(errorVar, EC_RaiseValue);
  }

  if (errorExpr) {
    // If we had an error, emit it.
    emitter.emitExpr(errorExpr, dest);
  } else {
    // Figure it if we're in a try, and if so, which subregion.
    auto [tryOp, inExceptRegion] = findParentTry(builder.getInsertionBlock());

    // Otherwise, we must be in the 'except' part of the try block and are
    // rethrowing the current error.  This isn't correct Python semantics, see
    // the caveat above.
    if (!inExceptRegion) {
      emitError(loc.Start, "'raise' must live within an 'except' block or a "
                           "function marked 'raises'")
          << loc;
      dest.resetForError(emitter);
      return success();
    }

    // Re-raise the contextual exception.
    SyntheticNode synthNode(loc.Start);
    emitter.emitResult(MRValue(tryOp.getErr()), &synthNode, dest);
  }

  // If we are in a debug build, we inject a call to a stop hook for the
  // debugger right before a RaiseOp.
  if (shared.options.debugLevel !=
      CompilationOptions::DebugInfoLevel::kNoDebug) {
    if (failed(injectDebuggerRaiseHookCall(shared, emitter, getParentDecl(),
                                           loc.Start, errorExpr)))
      return failure();
  }

  if (inferringErrorType)
    emitter.checkInferredErrorType(errSlot.getRValueType(), loc.Start);

  LIT::RaiseOp::create(builder, translateLocation(loc.Start));
  return success();
}

/// assert_stmt ::= "assert" expression ["," expression]
///
/// Desugars into a call to debug_assert[assert_mode="safe"](cond, msg).
ParseResult StmtParser::parseAssertStmt(size_t assertIndent) {
  SMLoc kwLoc = consumeToken(Token::kw_assert).getLoc();

  // Must be inside a function body.
  if (!isa_and_nonnull<FnOp>(getParentDecl().getIfOperation())) {
    emitError(
        kwLoc,
        "'assert' must be inside a function; move this into a function body");
    return success();
  }

  // Parse the condition expression.
  ExprNode *condExpr = nullptr;
  if (parseExpression(condExpr, assertIndent))
    return failure();

  // Parse optional message expression.
  ExprNode *messageExpr = nullptr;
  if (consumeIf(Token::comma))
    if (parseExpression(messageExpr, assertIndent))
      return failure();

  // Look up the debug_assert function from the standard library.
  ArrayRef<ASTDecl *> debugAssertFns = shared.getBuiltinFunction(
      getDeclScope(), {"std", "builtin", "debug_assert"}, "debug_assert",
      kwLoc);
  if (debugAssertFns.empty())
    return success(); // Error already emitted.

  // Emit the condition expression as an AnyValue.
  auto emitter = getEmitter();
  AnyValue condVal = emitter.emitExpr(condExpr, EC_TopLevelStmt);
  if (!condVal)
    return success();

  // Build the call operands: debug_assert(cond[, msg]).
  CallOperands operands(CallSyntax::kDirectCall, condExpr, EC_TopLevelStmt);
  operands.add({condVal, condExpr});
  if (messageExpr) {
    AnyValue msgVal = emitter.emitExpr(messageExpr, EC_TopLevelStmt);
    if (!msgVal)
      return success();
    operands.add({msgVal, messageExpr});
  }

  // Create the overload set and emit the call.
  ParamBindings bindings(getDeclScope(), condExpr);
  OverloadSet call("debug_assert", debugAssertFns, std::move(bindings),
                   CallSyntax::kDirectCall);
  call.emitCall(std::move(operands), emitter);
  return success();
}

/// break_stmt ::= "break"
/// continue_stmt ::= "continue"
ParseResult StmtParser::parseBreakOrContinueStmt(Token::Kind kind,
                                                 StringRef name,
                                                 StringRef opName) {
  llvm::SMLoc loc = consumeToken(kind).getLoc();

  // We diagnose break/continue that are not in a loop in LowerSemanticCF.

  // Split the block at the insertion point. Any subsequent statements are dead
  // code. Let region DCE handle it.
  OperationState state(translateLocation(loc), opName);
  builder.create(state);
  return success();
}

//===----------------------------------------------------------------------===//
// Compound statements.
//===----------------------------------------------------------------------===//

/// while_stmt ::=  "while" assignment_expression ":" suite
///                 ["else" ":" suite]
ParseResult StmtParser::parseWhileStmt(size_t curIndent) {
  Location whileLoc = translateLocation(consumeToken(Token::kw_while).getLoc());

  ExprNode *condExp = nullptr;
  if (parseExpression(condExp, curIndent, Precedence::kAssignExpr))
    return failure();

  // We will be moving the builder into sub-regions that are created, make sure
  // we end up after it when this is done.
  llvm::SaveAndRestore builderSaver(builder);

  // Create the LoopOp
  auto loopOp = LIT::LoopOp::create(builder, whileLoc);
  Block *bodyBlock = builder.createBlock(&loopOp.getBodyRegion());
  Block *elseBlock = builder.createBlock(&loopOp.getElseRegion());

  // Create the body region.
  builder.setInsertionPointToStart(bodyBlock);

  // Emit the condition expression into an if/break pattern:
  // if <cond>:
  //   hlcf.yield
  // else:
  //   lit.loop.break.else  // Jump to the 'else' block.
  auto emitter = getEmitter();
  RValue condRVal = emitter.emitExprScalarBool(condExp, EC_BoolCondition);
  Value condVal =
      emitter.emitSRValue({AnyValue(condRVal), condExp}, EC_BoolCondition);

  // After the condition is evaluated, validate the end of the statement.
  if (parseToken(Token::colon, "expected ':' after expression"))
    return failure();
  if (!condVal)
    return success(); // IRGen error already emitted; parse succeeded!

  Operation *condIf = emitter.emitIfThen(
      whileLoc, condVal, TypeRange{},
      [&]() -> LogicalResult {
        HLCF::YieldOp::create(*emitter.builder, whileLoc);
        return success();
      },
      [&]() -> LogicalResult {
        LoopBreakElseOp::create(*emitter.builder, whileLoc);
        return success();
      });
  if (!condIf)
    return success();
  builder.setInsertionPointAfter(condIf);

  if (failed(parseLocalScopeSuite(curIndent)))
    return failure();
  LIT::LoopContinueOp::create(builder, whileLoc);

  // Create the else region.
  builder.setInsertionPointToStart(elseBlock);
  // The 'else' block is executed only when the condition check fails.
  if (isTokenInCurrentStatement(curIndent, /*allowSameIndent=*/true) &&
      consumeIf(Token::kw_else)) {
    if (parseToken(Token::colon, "expected ':' after else") ||
        parseLocalScopeSuite(curIndent))
      return failure();
  }
  LIT::LoopYieldOp::create(builder, whileLoc);
  return success();
}

/// match_stmt ::=  "match" subject_expr ":" NEWLINE
///                 case_block+
/// case_block  ::= "case" pattern ["if" expression] ":" suite
///
ParseResult StmtParser::parseMatchStmt(size_t curIndent) {
  SMLoc matchLoc = consumeToken(Token::kw___match).getLoc();
  Location matchLocation = translateLocation(matchLoc);

  // Parse the match subject.
  ExprNode *subjectExpr = nullptr;
  if (parseExpression(subjectExpr, curIndent, Precedence::kAssignExpr) ||
      parseToken(Token::colon, "expected ':' after match subject"))
    return failure();

  // Evaluate the subject once; match IR will consume this later.
  CValue subject = getEmitter().emitExprCValue(subjectExpr, EC_MatchSubject);
  if (!subject) {
    // If we failed to emit the subject expression, skip over the body of the
    // match statement entirely. We do this by skipping any same-indent `case`
    // blocks that belong to this match, then stop at the next
    // same-or-less-indented token (this also covers Python-style cases
    // indented under the match).
    while (isTokenInCurrentStatement(curIndent, /*allowSameIndent=*/true) &&
           getToken().is(Token::kw_case)) {
      size_t caseIndent = getToken().getIndentation().value_or(curIndent);
      consumeToken(Token::kw_case);
      skipUntilIndentation(caseIndent);
    }
    skipUntilIndentation(curIndent);
    return success();
  }

  // Parse one or more case blocks. Cases may share the match indent (Mojo
  // style) or be indented beneath it (Python style).  We parse each of the
  // patterns before emitting the IR so we can optimize the pattern tests and
  // allocate the ElIf statement once.
  struct CaseEntry {
    ExprNode *patternExpr;
    ExprNode *guardExpr;
    LexerCursor caseCursor;
  };
  SmallVector<CaseEntry, 4> caseEntries;

  while (isTokenInCurrentStatement(curIndent, /*allowSameIndent=*/true) &&
         getToken().is(Token::kw_case)) {
    size_t caseIndent = getToken().getIndentation().value_or(curIndent);
    consumeToken(Token::kw_case);

    // Parse the case pattern as an expression. Stop before `if` so a trailing
    // match guard is not absorbed as a ternary `x if y else z`. `as` is looser
    // than `if`, so a top-level `case <pattern> as name` is attached here.
    ExprNode *patternExpr = nullptr;
    if (parseExpression(patternExpr, caseIndent,
                        Precedence(int(Precedence::kIfElse) + 1))) {
      skipUntilIndentation(caseIndent);
      continue;
    }

    // Optional `as` binding: `case <pattern> as name`. Nested `as` inside
    // parentheses is handled by the expression parser.
    SMLoc asLoc;
    if (consumeIf(Token::kw_as, &asLoc)) {
      Token nameTok = getToken();
      if (parseIdentifier("expected a name after 'as'")) {
        skipUntilIndentation(caseIndent);
        continue;
      }
      auto *name = shared.allocPersistent<DeclRefNode>(
          nameTok.getSpelling(), nameTok.is(Token::escaped_identifier));
      patternExpr = shared.allocPersistent<BinOpNode>(ExprNode::kAsPat,
                                                      patternExpr, asLoc, name);
    }

    // Optional match guard: `case <pattern> if <cond>:`.
    ExprNode *guardExpr = nullptr;
    if (consumeIf(Token::kw_if)) {
      if (parseExpression(guardExpr, caseIndent, Precedence::kAssignExpr)) {
        skipUntilIndentation(caseIndent);
        continue;
      }
    }
    if (parseToken(Token::colon, "expected ':' after case pattern")) {
      skipUntilIndentation(caseIndent);
      continue;
    }

    // Okay, we successfully parsed a case block. Remember it for later.
    caseEntries.push_back({patternExpr, guardExpr, getLexer().getCursor()});
    skipUntilIndentation(caseIndent);
  }

  if (caseEntries.empty()) {
    emitError(matchLoc) << "'__match' statement must have at least one 'case' "
                           "block";
    return success();
  }

  auto afterCaseCursor = getLexer().getCursor();

  // Given we have the pile of pattern collected together in a list, we can add
  // emission optimizations to improve the order various sub-patterns are
  // emitted.  For now, we simply emit each linearly.

  // Because we don't know whether a case is allowed to consume an RValue, we
  // convert the subject to a BValue, so none of the pattern emission can
  // consume the RValue.  For example, any "var" bindings will have to do a
  // copy.
  // TODO: maintain RValueness for as long as we can.
  BValue subjectBVal =
      getEmitter().emitBValue({subject, subjectExpr}, EC_MatchSubject);
  if (!subjectBVal)
    return failure();

  auto emitCasePredicate = [&](const CaseEntry &caseEntry) -> SRValue {
    auto emitter = getEmitter();
    CValue condCVal = caseEntry.patternExpr->emitMatch(emitter, subjectBVal,
                                                       PatternDeclKind::kNone);

    // If a guard predicate is present, AND it with the pattern predicate.
    // Always evaluate the guard even when the pattern is known-false so
    // diagnostics still fire for an unreachable guard.
    if (condCVal && caseEntry.guardExpr)
      condCVal = emitter.emitAndMatchPredicates(
          {condCVal, caseEntry.patternExpr},
          [&]() -> ASTExprAnd<CValue> {
            return {emitter.emitExprScalarBool(caseEntry.guardExpr,
                                               EC_BoolCondition),
                    caseEntry.guardExpr};
          },
          /*evaluateRhsEvenIfLhsFalse=*/true);
    return emitter.emitSRValue({condCVal, caseEntry.patternExpr},
                               EC_BoolCondition);
  };

  auto emitCaseBody = [&](Block &bodyBlock, const CaseEntry &caseEntry,
                          Location caseLoc) -> ParseResult {
    builder.setInsertionPointToStart(&bodyBlock);
    // Change the parser cursor to the start of the case body so we can parse
    // the right text. Use parseSuite (not parseLocalScopeSuite) so the body
    // shares the case scope above — pattern bindings remain visible.
    caseEntry.caseCursor.restore(getLexer());
    if (failed(parseSuite(curIndent)))
      return failure();
    HLCF::YieldOp::create(builder, caseLoc);
    return success();
  };

  // First case: pattern bindings are created while emitting the predicate.
  // Keep them in a case-local scope that also wraps the body. The predicate
  // itself becomes the elif operand; later cases stay in-region for
  // short-circuit.
  HLCF::ElifOp elifOp;
  {
    DebugInfo::DIBuilder::ScopeGuard scopeGuard;
    llvm::SaveAndRestore<ASTDecl *> keepDecl(curDeclScope);
    pushChildScope(scopeGuard, keepDecl);

    auto firstCaseLoc =
        translateLocation(caseEntries.front().patternExpr->getLoc());
    SRValue firstCond = emitCasePredicate(caseEntries.front());

    // A bad first pattern still needs an operand for the elif, so recover
    // with a statically-false arm. That keeps the later cases (and the rest
    // of the enclosing suite) parsed, so their diagnostics still fire.
    Value condValue =
        firstCond ? Value(firstCond)
                  : ParamConstantOp::create(
                        builder, firstCaseLoc,
                        SIMDAttr::getScalarBool(builder.getContext(), false));

    unsigned numExtraRegions =
        caseEntries.size() > 1 ? (caseEntries.size() - 1) * 2 : 0;
    elifOp = HLCF::ElifOp::create(builder, matchLocation, TypeRange(),
                                  condValue, numExtraRegions);
    elifOp.getThenRegion().emplaceBlock();
    elifOp.getElseRegion().emplaceBlock();
    for (Region &region : elifOp.getElifRegions())
      region.emplaceBlock();

    if (!firstCond) {
      builder.setInsertionPointToStart(&elifOp.getThenRegion().front());
      HLCF::YieldOp::create(builder, firstCaseLoc);
    } else if (failed(emitCaseBody(elifOp.getThenRegion().front(),
                                   caseEntries.front(), firstCaseLoc))) {
      return failure();
    }
  }

  for (auto [idx, caseEntry] :
       llvm::enumerate(ArrayRef<CaseEntry>(caseEntries).drop_front())) {
    unsigned extraIdx = idx; // 0-based into additional (cond, then) pairs
    auto &condBlock = elifOp.getElifRegions()[extraIdx * 2].front();
    builder.setInsertionPointToStart(&condBlock);

    DebugInfo::DIBuilder::ScopeGuard scopeGuard;
    llvm::SaveAndRestore<ASTDecl *> keepDecl(curDeclScope);
    pushChildScope(scopeGuard, keepDecl);

    auto caseLoc = translateLocation(caseEntry.patternExpr->getLoc());
    SRValue condRVal = emitCasePredicate(caseEntry);
    if (!condRVal)
      continue;
    HLCF::ElifYieldOp::create(builder, caseLoc, condRVal,
                              /*no extra values*/ ValueRange());

    if (failed(emitCaseBody(elifOp.getElifRegions()[extraIdx * 2 + 1].front(),
                            caseEntry, caseLoc)))
      return failure();
  }

  // The "else" of the elif is a noop.  TODO: We should mark this as unreachable
  // if there is an irrefutable pattern so we don't get dead code errors.
  builder.setInsertionPointToStart(&elifOp.getElseRegion().front());
  HLCF::YieldOp::create(builder, matchLocation);
  builder.setInsertionPointAfter(elifOp);
  afterCaseCursor.restore(getLexer());
  return success();
}

/// for_stmt ::=  "for" target_list "in" starred_list ":" suite
///              ["else" ":" suite]
ParseResult StmtParser::parseForStmt(LexerCursor startCursor,
                                     size_t curIndent) {
  SMLoc forLoc;
  ExprNode *targetExpr = nullptr;
  ExprNode *seqExpr = nullptr;
  if (parseForTargetAndSequence(curIndent, forLoc, targetExpr, seqExpr))
    return failure();

  // We will be moving the builder into sub-regions that are created, make sure
  // we end up after it when this is done.
  llvm::SaveAndRestore builderSaver(builder);

  // Otherwise, this is a dynamic for stmt.
  auto forStmt = emitForStmt(
      forLoc, targetExpr, seqExpr,
      [&]() -> LogicalResult { return parseSuite(curIndent); },
      [&]() { skipUntilIndentation(curIndent); });

  if (!forStmt) {
    // If the error happened in the loop body, we stop parsing completely.
    // If the error happened in the for statement, report it and keep gathering
    // more errors from the parent suite
    return failure(forStmt.hasErrorInLoopBody());
  }

  LIT::LoopOp loopOp = forStmt.getLoopOp();

  // The 'else' block is executed only when the condition check fails.
  if (isTokenInCurrentStatement(curIndent, /*allowSameIndent=*/true) &&
      consumeIf(Token::kw_else)) {
    builder.setInsertionPointToStart(&loopOp.getElseRegion().front());
    if (parseToken(Token::colon, "expected ':' after else") ||
        parseLocalScopeSuite(curIndent))
      return failure();
  }

  return success();
}

// This emits the pattern for a 'for' loop, calling the specified 'bodyFn'
// closure on success when in the scope of the loop, and the specified
// 'errorFn' if there is a semantic error with the sequence expression or
// target.
// Return a struct with parsed ForOp or error kind to either abort parsing of
// the parent suite or keep parsing it to get more errors.
LoopResult StmtParser::emitForStmt(SMLoc forLoc, ExprNode *targetExpr,
                                   ExprNode *seqExpr,
                                   std::function<LogicalResult()> bodyFn,
                                   std::function<void()> errorFn) {
  // We will be moving the builder into sub-regions that are created, make sure
  // we end up after it when this is done.
  llvm::SaveAndRestore builderSaver(builder);

  Location forLocation = translateLocation(forLoc);

  // If there is a failure before we parse the for loop body, we still want to
  // call the parser on it so that it builds an ASTDecl node and adds the for
  // loop VarDecl to the lookup path.  Otherwise, we will get spurious “use of
  // unknown declaration” errors on it besides whatever error is raised while
  // processing the loop header.
  auto skipBodyOnFailure = llvm::scope_exit([&]() {
    if (errorFn)
      errorFn();
  });

  // We desugar
  //
  //   for e in iterable:
  //     <BODY>
  //
  // into:
  //   var $ITER = iterable.__iter__()
  //   while True:
  //       ref e / var e
  //       try:
  //           e = $ITER.__next__()
  //       except:
  //           break
  //       <BODY>
  //
  // or:
  //   var $ITER = iterable.__iter__()
  //   while $ITER.__has_next__():
  //       ref e = $ITER.__next__()
  //       <BODY>
  auto prefixEmitter = getEmitter();

  // Emit the expression for the iterable.
  ASTExprAnd<AnyValue> loadedSeq = {
      prefixEmitter.emitExpr(seqExpr, EC_ForIterator), seqExpr};
  if (!loadedSeq.ir)
    return LoopResult(LoopResult::ErrorKind::inLoopStmt);

  // Get an iterator into the iterable by emitting a call to `__iter__`.
  VarDeclOp iterVar =
      prefixEmitter.emitVarDecl("$ITER", UnresolvedType::get(getContext()),
                                forLocation, VarDeclKind::Synthesized);
  ExprDest rangeDest(iterVar, EC_ForIterator);
  if (!prefixEmitter.emitNamedMethodCall(
          "__iter__", CallOperands(CallSyntax::kMethodCall, seqExpr,
                                   std::move(rangeDest), {{loadedSeq}})))
    return LoopResult(LoopResult::ErrorKind::inLoopStmt);

  // Emit the call to __next__ with the target as the destination to assign
  // into.  This will synthesize the VarDeclOp from the inferred result type,
  // which will be in scope for the body that we will parse.
  ExprDest indvarDest(targetExpr, EC_ForIterator);
  // Lexically scope the indvarDest as a 'bind' pattern like an 'imm' arg.
  indvarDest.setPatternDeclKind(PatternDeclKind::kBind);

  // Now that we have the iterator (and its type), find the  __next__ method to
  // know what we're dealing with.
  CallOperands nextOperands(CallSyntax::kMethodCall, seqExpr,
                            std::move(indvarDest),
                            {{MLValue(iterVar), seqExpr}});
  ASTType iterType = iterVar.getType().getElementType();

  PValue nextFn = OverloadSet::lookupAndResolve(iterType, "__next__",
                                                nextOperands, prefixEmitter);
  if (!nextFn) {
    auto diag = emitError(seqExpr->getLoc());
    diag << iterType
         << " does not conform to 'Iterable'; add conformance to use in a "
            "'for' loop"
         << seqExpr->getRange();
    diag.attachNote(seqExpr->getLoc())
        << "to conform to 'Iterable', add it to the struct declaration: "
           "'struct Foo(Iterable):'";
    nextOperands.dest.resetForError(prefixEmitter);
    return LoopResult(LoopResult::ErrorKind::inLoopStmt);
  }

  // Determine if we're modern or legacy structure.
  bool isThrowsCase =
      FnOrFnLiteralTypeGeneratorType::get(nextFn.getType()).isThrows();

  // Create the LoopOp
  auto loopOp = LIT::LoopOp::create(builder, forLocation);
  // Start with a noop 'else' region.
  (void)builder.createBlock(&loopOp.getElseRegion());
  LIT::LoopYieldOp::create(builder, forLocation);

  // Create the body.
  builder.createBlock(&loopOp.getBodyRegion());

  // Create the body. Add Target element to the continue block by calling next
  // method. Emit the result into an implicitly declared variable at the current
  // scope.

  // Push a new local variable scope for the subsequent body.
  DebugInfo::DIBuilder::ScopeGuard scopeGuard;
  llvm::SaveAndRestore<ASTDecl *> keepDecl(curDeclScope);
  pushChildScope(scopeGuard, keepDecl);
  auto emitter = getEmitter();

  if (isThrowsCase) {
    // Call __next__.  If it throws then break to the else block.
    if (!emitter.emitIndirectCallInTryBlock(
            nextFn, std::move(nextOperands), [&](VarDeclOp errDecl) {
              // Just break on error.  We ignore the actual error value.
              LoopBreakElseOp::create(*emitter.builder, loopOp.getLoc());
            })) {
      indvarDest.resetForError(emitter);
      return LoopResult(LoopResult::ErrorKind::inLoopStmt);
    }
  } else {
    CValue hasNextBool = emitter.emitNamedMethodCall(
        "__has_next__",
        CallOperands(CallSyntax::kMethodCall, seqExpr, EC_ForIterator,
                     {{MLValue(iterVar), seqExpr}}));
    CValue hasNext =
        emitter.emitScalarBool({hasNextBool, seqExpr}, EC_ForIterator);
    SRValue shouldContinue =
        emitter.emitSRValue({hasNext, seqExpr}, EC_ForIterator);
    if (!shouldContinue) {
      indvarDest.resetForError(emitter);
      return LoopResult(LoopResult::ErrorKind::inLoopStmt);
    }

    // Emit an if statement, if the condition is true then yield other break to
    // the else block.
    Operation *condIf = emitter.emitIfThen(
        forLocation, shouldContinue, TypeRange{},
        [&]() -> LogicalResult {
          HLCF::YieldOp::create(*emitter.builder, forLocation);
          return success();
        },
        [&]() -> LogicalResult {
          LoopBreakElseOp::create(*emitter.builder, forLocation);
          return success();
        });
    if (!condIf) {
      indvarDest.resetForError(emitter);
      return LoopResult(LoopResult::ErrorKind::inLoopStmt);
    }
    builder.setInsertionPointAfter(condIf);
    emitter.builder = builder;

    // Emit the call to __next__ now that we know there is an element.
    if (!emitter.emitIndirectCall(nextFn, std::move(nextOperands))) {
      indvarDest.resetForError(emitter);
      return LoopResult(LoopResult::ErrorKind::inLoopStmt);
    }
  }

  // We're parsing the body at this point, tell the caller not to skip it.
  skipBodyOnFailure.release();

  // Parse the body of the for loop, using the callback.
  if (failed(bodyFn()))
    return LoopResult(LoopResult::ErrorKind::inLoopBody);
  LIT::LoopContinueOp::create(builder, forLocation);
  return LoopResult(loopOp);
}

// FIXME: This needs to parse this as a target expression and then handle it
// like a destructuring pattern.
static StringAttr decodeTarget(ExprNode *targetExpr, SharedState &shared) {
  StringRef name;
  if (targetExpr->kind == ExprNode::kDiscardLiteral)
    name = "_";
  else if (auto dre = dyn_cast<DeclRefNode>(targetExpr))
    name = dre->spelling;
  else {
    shared.emitError(
        targetExpr->getLoc(),
        "'for' loop variables must be identifiers; use a valid name");
    return {};
  }
  return StringAttr::get(shared.getContext(), name);
}

ParseResult StmtParser::parseParamFor(size_t curIndent, SMLoc forLoc,
                                      ExprNode *targetExpr, ExprNode *seqExpr) {
  Location forLocation = translateLocation(forLoc);
  ASTDecl &scope = getParentDecl();

  // All expressions we need are emitted into the param domain.
  IREmitter emitter = getParamEmitter(EC_ForIterator);

  // On any semantic failure, skip the body for better error recovery.
  auto skipBodyOnFailure =
      llvm::scope_exit([&]() { skipUntilIndentation(curIndent); });

  // For loops generally desugar into:
  //   var it = iterable.__iter__()
  //   while True:
  //       var e =
  //          try:
  //              yield it.__next__()  # Throws StopIteration if no more
  //              elements.
  //          except StopIteration:
  //              lit.loop.break.else
  //       <BODY>
  // We capture the "it" expression as a PValue and the "__next__" callees so we
  // can iterate the value in the elaborator.  However, the elaborator struggles
  // with 'mut' arguments and exceptions, so we use paramfor_has_next,
  // paramfor_next_iter, and paramfor_next_value functions to 'functional'ize.
  //
  // Parameter for loops are desugared into:
  //   kgen.param.for 'it', initial=iterable.__iter__(),
  //      has_next=..., get_next=... {
  //       comptime if it.has_next():
  //         # Logically: alias e = it.__next__()
  //         alias e = paramfor_next_value(it)
  //         <BODY>
  //       else:
  //          param.for.else
  //
  // The elaborator instantiates the body of the loop N times with different
  // versions of the iterator.

  // Emit the sequence and call __iter__ on it.
  AnyValue seqValue = emitter.emitExprCValue(seqExpr, EC_ComptimeForSeq);
  if (!seqValue)
    return failure();
  PValue initialIterVal = emitter.emitPValue(
      {emitter.emitNamedMethodCall(
           "__iter__", CallOperands(CallSyntax::kMethodCall, seqExpr,
                                    EC_ForIterator, {{seqValue, seqExpr}})),
       seqExpr},
      EC_ForIterator);
  if (!initialIterVal)
    return failure();

  ASTType iterType = initialIterVal.getRValueType();

  // Resolve the paramfor_next_iter(iterator) and paramfor_next_value(iterator)
  // functions.
  auto getMutFnWrapper = [&](StringRef name) -> PValue {
    // Bind the sequence initial value to the parameter for iterator generator.
    // Start by looking up the builtin generator.
    ArrayRef<ASTDecl *> paramForImpl = shared.getBuiltinFunction(
        scope, {"std", "builtin", "_stubs"}, name, forLoc);
    if (paramForImpl.empty())
      return {};

    // Resolve the overload with the sequence's type. This succeeds if the
    // iterator type is currently supported.
    ParamBindings bindings(scope, seqExpr);
    bindings.add(seqExpr, PValue(iterType));
    // We use paramfor_next_iter as a wrapper because the elaborator doesn't
    // have a strong enough memory model to handle "mut" arguments to next.
    OverloadSet call(name, paramForImpl, std::move(bindings),
                     CallSyntax::kDirectCall);
    PValue literal = call.getDirectSymbol(/*expectedType=*/{},
                                          emitter.deferredTypingContext);
    // Must resolved to a function literal. Extract the literal target.
    return sugarCast<FnLiteralTypeGeneratorType>(literal.getType())
        .getSymbolConstantAttr();
  };

  PValue hasNext = getMutFnWrapper("paramfor_has_next");
  if (!hasNext)
    return failure();
  PValue getNextIter = getMutFnWrapper("paramfor_next_iter");
  if (!getNextIter)
    return failure();
  PValue getNextValue = getMutFnWrapper("paramfor_next_value");
  if (!getNextValue)
    return failure();

  // Build the iterator value parameter.
  auto iterDecl = ParamDeclAttr::get(scope.mangleParamName("iter"), iterType);

  // Create the loop and parse the body into it.
  auto paramFor = ParamForOp::create(builder, forLocation, initialIterVal,
                                     hasNext, getNextIter, iterDecl);

  builder.createBlock(&paramFor.getBody());

  // The entry to the body should be a check for the end of sequence:
  //  comptime if !iter.has_next(). Emit the condition as a parameter
  // expression.
  auto iterValue = PValue(ParamDeclRefAttr::get(iterDecl));
  CValue hasNextRes = emitter.emitIndirectCall(
      hasNext, CallOperands(CallSyntax::kMethodCall, seqExpr, EC_ForIterator,
                            {{iterValue, seqExpr}}));
  auto hasNextBool =
      emitter.emitScalarBool({hasNextRes, seqExpr}, EC_ForIterator);
  if (!hasNextBool)
    return failure();
  assert(hasNextBool.getIfPValue() && "expected PValue in param context");
  auto paramIf =
      ParamIfOp::create(builder, forLocation, hasNextBool.getIfPValue());

  // Keep going if we have more elements.
  builder.createBlock(&paramIf.getThenRegion());

  // If not, go to the else block.
  builder.createBlock(&paramIf.getElseRegion());
  ParamForGotoElseOp::create(builder, forLocation);
  // Keep inserting after this operation.
  builder.setInsertionPointAfter(paramIf);
  // We always continue or goto-else from the arms of the param.if.
  UnreachableOp::create(builder, forLocation);

  // After the check for too-few elements, we extract the next element and bind
  // to the target by calling the paramfor_next_iter "next_value" function.
  auto nextValue = emitter.emitIndirectCall(
      getNextValue, CallOperands(CallSyntax::kDirectCall, seqExpr,
                                 EC_ForIterator, {{iterValue, seqExpr}}));
  if (!nextValue)
    return failure();
  assert(nextValue.getIfPValue() && "expected PValue in param context");

  // Everything resolved, so we'll be able to parse the body, don't skip it.
  skipBodyOnFailure.release();

  { // Create a scope for the induction variable bindings.
    DebugInfo::DIBuilder::ScopeGuard scopeGuard;
    llvm::SaveAndRestore<ASTDecl *> keepDecl(curDeclScope);
    // Push a new local variable scope for the subsequent suite.
    pushChildScope(scopeGuard, keepDecl);

    IREmitter emitter = getParamEmitter(EC_ForIterator);
    if (failed(emitter.emitDestructuringPValue(nextValue.getIfPValue(),
                                               targetExpr)))
      return failure();

    //  Parse into the 'then' region of the parameter if.
    builder.setInsertionPointToStart(&paramIf.getThenRegion().front());
    // Parse the body.
    if (parseSuite(curIndent))
      return failure();
    ParamForContinueOp::create(builder, forLocation);
  }

  // Parse the else region if present.
  builder.createBlock(&paramFor.getElseRegion());
  // The 'else' block is executed only when the condition check fails.
  if (isTokenInCurrentStatement(curIndent, /*allowSameIndent=*/true) &&
      consumeIf(Token::kw_else)) {
    if (parseToken(Token::colon, "expected ':' after else") ||
        parseLocalScopeSuite(curIndent))
      return failure();
  }
  ParamYieldOp::create(builder, forLocation);

  // Advance the insertion point.
  builder.setInsertionPointAfter(paramFor);
  return success();
}

/// The finally block executes whenever control flow leaves any of the other
/// regions of a `try`, whether through a yield, return, or raise. Its overall
/// control flow effect takes precedence over how control flow left the other
/// `try` regions originally.
///
/// This is conceptually implemented by branching before said exits to the
/// finally region, and when yielding from the finally region, branch back to
/// where control was before.
///
/// However, this means that the finally region can conditionally overwrite an
/// error slot and then choose not to raise, causing this issue:
///
/// ```
/// def raising_finally():
///     try:
///         raise Error() # initializes %__error__, then branch to 'finally'
///     finally:
///         # conservatively destroy %__error__ before the raising call
///         might_raise()
///         # if the call didn't raise, branch back to where we were in 'try',
///         # but now we have an error return with an uninitialized %__error__!
/// ```
///
/// Thus, the error slot in the finally region cannot alias the one used in the
/// other regions, because it might conditionally overwrite it while it still
/// needs to be used. Fix this by rewriting the above into:
///
/// ```
/// def raising_finally():
///     try:
///         raise Error()
///     finally:
///         try:
///             might_raise()
///         except:
///             raise
/// ```
ParseResult StmtParser::handleRaisingFinallyRegion(
    TryOp tryOp, SMLoc loc, function_ref<ParseResult()> populateFinallyBody) {
  if (tryOp.hasTrivialFinally())
    return success();

  MLValue errSlot = getEmitter().findNearestErrorSlot();
  if (!errSlot)
    return populateFinallyBody();
  bool inferringErrorType = isa<UnresolvedType>(errSlot.getRValueType());

  VarDeclOp errDecl = getEmitter().emitVarDecl(
      "__finally_error__", UnresolvedType::get(getContext()), tryOp.getLoc(),
      VarDeclKind::Synthesized);
  auto nestedTry = TryOp::create(builder, tryOp.getLoc(), errDecl,
                                 /*suppressWarnings=*/true);

  // Stub out the else and finally regions of this try.
  builder.createBlock(&nestedTry.getElseRegion());
  TryYieldOp::create(builder, tryOp.getLoc());
  builder.createBlock(&nestedTry.getFinallyRegion());
  TryYieldOp::create(builder, tryOp.getLoc());

  Block *tryBlock = builder.createBlock(&nestedTry.getTryRegion());
  if (populateFinallyBody())
    return failure();
  builder.setInsertionPointToEnd(tryBlock);
  TryYieldOp::create(builder, tryOp.getLoc());

  // Move the error into the overall error slot.
  builder.createBlock(&nestedTry.getExceptRegion());

  // If no errors got emitted, then the catch block is unreachable.
  if (isa<UnresolvedType>(errDecl.getType().getElementType())) {
    UnreachableOp::create(builder, tryOp.getLoc());
  } else {
    ExprDest moveDest(errSlot, EC_RaiseValue);
    if (isa<UnresolvedType>(errSlot.getRValueType())) {
      auto errorVar = cast<VarDeclOp>(errSlot.getDefiningOp());
      moveDest = ExprDest(errorVar, EC_RaiseValue);
    }
    SyntheticNode synthNode(loc);
    getEmitter().emitResult(MRValue(errDecl), &synthNode, moveDest);
    if (inferringErrorType)
      getEmitter().checkInferredErrorType(errSlot.getRValueType(), loc);
    RaiseOp::create(builder, tryOp.getLoc());
    TryYieldOp::create(builder, tryOp.getLoc());
  }

  builder.setInsertionPointAfter(nestedTry);
  return success();
}

/// try_stmt ::= "try" ":" suite "except" [expression ["as" identifier]] ":"
///              suite ["else" suite]
ParseResult StmtParser::parseTryStmt(size_t curIndent) {
  SMLoc smLoc = consumeToken(Token::kw_try).getLoc();
  Location loc = translateLocation(smLoc);

  if (parseToken(Token::colon, "expected ':' after 'try'"))
    return failure();

  // If we see a 'try' block in a context that cannot raise, we need to check if
  // the user explicitly provided an 'except' region, otherwise this is a
  // try-finally block where the try block cannot raise.
  bool inExceptRegion = !!getEmitter().findNearestErrorSlot();
  if (!inExceptRegion) {
    Lexer subLexer(shared.diags, lexer.getCursor());
    ParserBase subParser(shared, subLexer);
    subParser.skipUntilIndentation(curIndent);
    inExceptRegion = subParser.consumeIf(Token::kw_except);
  }

  // Restore the builder to its current insertion point after parsing.
  llvm::SaveAndRestore builderSaver(builder);

  // We start the caught exception type as 'UnresolvedType' to allow throws
  // within the try body to infer this.  If there are no throwing operations
  // in the body, this will be left as-is, but won't be used.
  VarDeclOp errDecl = getEmitter().emitVarDecl(
      "__try_error__", UnresolvedType::get(getContext()), loc,
      VarDeclKind::Synthesized);
  auto tryOp = TryOp::create(builder, loc, errDecl);
  if (!inExceptRegion) {
    builder.createBlock(&tryOp.getExceptRegion());
    UnreachableOp::create(builder, loc);
  }

  // Parse the try suite.
  builder.createBlock(&tryOp.getTryRegion());
  if (parseLocalScopeSuite(curIndent))
    return failure();
  TryYieldOp::create(builder, translateLocation(getToken().getLoc()));

  // If nothing in the try body raised, the error vardecl may still be
  // unresolved.  Force it to Error type if so, so any uses of it complain about
  // Error.
  if (isa<UnresolvedType>(errDecl.getType().getElementType())) {
    if (auto errorType =
            shared.lookupBuiltinType("Error", getParentDecl(), smLoc))
      errDecl.changeElementType(errorType);
  }

  bool hasFinally = false;
  if (consumeIf(Token::kw_except)) {
    // Parse an optional target to bind the error.
    // FIXME: Our behavior is completely wrong here. "except Kind" is supposed
    // to be a type pattern, and we should be parsing it as such. You need to
    // use "except Kind as name" to bind the error to a name.
    ExprNode *errExpr = nullptr;
    if (getToken().isNot(Token::colon)) {
      if (parseTargetListExpr(errExpr, curIndent))
        return failure();
      errDecl->setLoc(translateLocation(errExpr->getLoc()));
    }

    if (parseToken(Token::colon, "expected ':' after 'except'"))
      return failure();

    builder.createBlock(&tryOp.getExceptRegion());

    // Parse the except suite into its own scope.
    {
      DebugInfo::DIBuilder::ScopeGuard scopeGuard;
      llvm::SaveAndRestore<ASTDecl *> keepDecl(curDeclScope);
      // Push a new local variable scope for the subsequent suite.
      pushChildScope(scopeGuard, keepDecl);

      // If an identifier was declared for the error value, add a declaration
      // that references it.
      if (errExpr) {
        if (StringAttr errName = decodeTarget(errExpr, shared)) {
          // If the user bound the error to a name, adjust the vardecl and add
          // the declaration.
          errDecl.setName(errName);
          errDecl.setKind(VarDeclKind::Var);

          // Add an ASTDecl for the error name. TODO: Generalize to an
          // arbitrary pattern.
          auto &vd = getDeclResolver().addFullyResolvedDecl(
              DeclIRValue(errDecl), errDecl.getNameAttr(), errExpr->getLoc(),
              curDeclScope);
          getEmitter().shared.notifyListenerOnVariableDecl(vd,
                                                           errExpr->getLoc());
        }
      }

      // Forward to the normal suite parse method.
      if (parseSuite(curIndent))
        return failure();
    }
    TryYieldOp::create(builder, translateLocation(getToken().getLoc()));

    // Parse the else suite if present. Otherwise, leave it as empty.
    builder.createBlock(&tryOp.getElseRegion());
    if (isTokenInCurrentStatement(curIndent, /*allowSameIndent=*/true) &&
        consumeIf(Token::kw_else)) {
      if (parseToken(Token::colon, "expected ':' after 'else'") ||
          parseLocalScopeSuite(curIndent))
        return failure();
    }
    TryYieldOp::create(builder, translateLocation(getToken().getLoc()));

    hasFinally = consumeIf(Token::kw_finally);
  } else {
    SMLoc finallyLoc;
    hasFinally = consumeIf(Token::kw_finally, &finallyLoc);
    if (!hasFinally)
      return emitTokenError("expected 'except' or 'finally' block");
    // In a raising context, the default 'except' block just forwards the error.
    if (inExceptRegion) {
      builder.createBlock(&tryOp.getExceptRegion());
      MLValue errSlot = getEmitter().findNearestErrorSlot();
      ExprDest dest(errSlot, EC_RaiseValue);

      // If the contextual caught type is unresolved, then we're the first raise
      // in a try block.  Resolve the error type to whatever we are raising.
      if (isa<UnresolvedType>(errSlot.getRValueType())) {
        auto errorVar = cast<VarDeclOp>(errSlot.getDefiningOp());
        dest = ExprDest(errorVar, EC_RaiseValue);
        getEmitter().checkInferredErrorType(errDecl.getType().getElementType(),
                                            finallyLoc);
      }
      SyntheticNode node(smLoc);
      getEmitter().emitResult(MRValue(errDecl), &node, dest);
      LIT::RaiseOp::create(builder, loc);
      TryYieldOp::create(builder, loc);
    }

    // Stub the 'else' region.
    builder.createBlock(&tryOp.getElseRegion());
    TryYieldOp::create(builder, loc);
  }
  builder.createBlock(&tryOp.getFinallyRegion());
  if (hasFinally) {
    if (handleRaisingFinallyRegion(tryOp, smLoc, [&] {
          if (parseToken(Token::colon, "expected ':' after 'finally'") ||
              parseLocalScopeSuite(curIndent))
            return failure();
          return mlir::success();
        }))
      return failure();
  }
  TryYieldOp::create(builder, loc);

  return success();
}

/// with_stmt ::=
///    "with" ( "(" with_stmt_contents ","? ")" | with_stmt_contents ) ":" suite
/// with_stmt_contents ::=  with_item ("," with_item)*
/// with_item          ::=  expression ["as" target]
ParseResult StmtParser::parseWithStmt(size_t curIndent) {
  SMLoc smLoc = consumeToken(Token::kw_with).getLoc();
  Location loc = shared.translateLocation(smLoc);

  return parseSingleWithStmt(curIndent, smLoc, loc);
}

/// Parses a single clause in the `with` statement, and possibly the body as
/// well.
/// This could recurse if there are multiple clauses in the `with` statement,
/// like:
///     with MyClass() as a, MyClass() as b:
///         ...
/// In that case, it interprets it as multiple nested "single" with statements,
/// like:
///     with MyClass() as a:
///         with MyClass() as b:
///             ...
/// This function handles just the `MyClass() as a`, then for everything
/// afterward it either recurses (for other clauses) or calls out to
/// `parseLocalScopeSuite` (for the body).
ParseResult StmtParser::parseSingleWithStmt(size_t curIndent, SMLoc smLoc,
                                            Location loc) {
  // With statements are just sugar for other constructs.  We desugar this:
  //     with EXPRESSION as TARGET:
  //       SUITE
  // Into:
  //     contextMgr = EXPRESSION
  //     TARGET = contextMgr.__enter__()
  //     try {
  //       SUITE
  //     } except(errorVal : Error) {
  //       hlcf.elif (contextMgr.__exit__(errorVal)) {
  //         hlcf.yield
  //       } else {
  //         raise errorVal
  //       }
  //       try.yield
  //     } else {
  //       contextMgr.__exit__()
  //     }
  // We elide the try and except logic when in a context that doesn't support
  // raising an error (like a non-raising fn).

  // Parse and emit the context mgr.
  ExprNode *contextExp = nullptr;
  if (parseExpression(contextExp))
    return failure();

  // Emit the context manager expression into a var with an inferred type.
  VarDeclOp contextMgrDecl = getEmitter().emitVarDecl(
      "$CONTEXTMGR", UnresolvedType::get(getContext()),
      shared.translateLocation(contextExp->getLoc()), VarDeclKind::Synthesized);
  ExprDest contextMgrDest(contextMgrDecl, EC_WithContextMgr);
  if (!getEmitter().emitExpr(contextExp, contextMgrDest))
    return failure();

  // Determine if the context manager has an __exit__ method.  If not, that is
  // fine, we silently just don't call it.  This mode of supporting context
  // managers with just an __enter__ method is useful for strong Mojo types
  // working with context managers even if they don't need them, e.g. we want
  // file descriptors to support both of these patterns:
  //
  //    with open("foo.txt", "r") as f:
  //        print(f.read())
  //
  // and:
  //    f = open("foo.txt", "r")
  //    print(f.read())
  //
  // The latter works because of Mojo's strong early-destruction guarantees and
  // lack of frame-objects-capturing-variables problems, but the former is more
  // familiar to Pythonistas.
  ASTType contextRVType = MLValue(contextMgrDecl).getRValueType();
  bool hasExitMethod =
      shared.typeHasMember(contextRVType, "__exit__", contextExp->getLoc());

  // Determine whether we're in a region that is allowed to raise.  If so,
  // generate logic to deal with it.
  MLValue errSlot = getEmitter().findNearestErrorSlot();
  bool inExceptRegion = !!errSlot;

  // If this has a 'as TARGET' specifier, parse the name into targetName,
  // otherwise targetName will be null.
  ExprNode *targetExpr = nullptr;
  if (consumeIf(Token::kw_as)) {
    if (parseExpression(targetExpr, curIndent, Precedence::kVarRefPat))
      return failure();
  }

  // We are about to generate the call to __enter__ but need to decide how to
  // pass the context expression, either as an LValue referring to the bound
  // variable, or as a transferred RValue if it takes it owned (enabling some
  // advanced use cases with unique context managers).
  AnyValue contextVal = MLValue(contextMgrDecl);

  // If there is an explicit target specified, use it.
  ExprDest enterDest(EC_WithContextMgr);
  if (targetExpr) {
    // Initialize the target expression with the result of the __enter__ call.
    enterDest = ExprDest(targetExpr, EC_WithContextMgr);
    // Bind a mutable target variable.
    enterDest.setPatternDeclKind(PatternDeclKind::kVar);
  }

  // Interrogate the caller to see what convention the first argument to the
  // __enter__ method is.  Be careful about invalid cases - the errors will get
  // diagnosed when emitting the method call.
  CallOperands enterOperands(CallSyntax::kMethodCall, contextExp,
                             std::move(enterDest));
  enterOperands.addSelf({contextVal, contextExp});
  auto enterEmitter = getEmitter();
  if (PValue enterMethod = OverloadSet::lookupAndResolve(
          contextRVType, "__enter__", enterOperands, enterEmitter)) {
    // If there is no exit method, we can pass the argument as an RValue so the
    // enter method can consume the value... unless __enter__ takes self 'mut'.
    auto sig = FnOrFnLiteralTypeGeneratorType::tryGet(enterMethod.getType());
    if (sig.has_value() && !sig->getArgConventions().empty()) {
      auto firstArgConvention = sig->getArgConventions()[0];
      if (firstArgConvention != ArgConvention::Mut && !hasExitMethod)
        contextVal = MRValue(contextMgrDecl);

      // One error that people hit is defining a context manager with both an
      // owned enter method and an exit method.  This will generate a terrible
      // error message in CheckLifetimes, so cut that off here.
      assert(firstArgConvention != ArgConvention::OwnedReg &&
             "not used by the mojo parser");
      if ((firstArgConvention == ArgConvention::OwnedMem ||
           firstArgConvention == ArgConvention::DeinitMem) &&
          hasExitMethod) {
        auto diag =
            emitError(contextExp->getLoc(), "context manager of type ")
            << contextRVType
            << " defines a consuming __enter__ method as well as an __exit__ "
               "method; either remove 'var' from its '__enter__' method or "
               "remove the '__exit__' method"
            << contextExp->getRange();
        if (ASTDecl *contextDecl = contextRVType.getDecl(shared))
          diag.attachNote(contextDecl->getLoc())
              << contextRVType << " declared here";

        // Make the emission work even if the type isn't copyable.
        contextVal = MRValue(contextMgrDecl);
      }
      enterOperands[0].ir = contextVal;
    }
  }

  DebugInfo::DIBuilder::ScopeGuard scopeGuard;
  llvm::SaveAndRestore<ASTDecl *> keepDecl(curDeclScope);
  pushChildScope(scopeGuard, keepDecl);

  // Emit the call to __enter__ and (if 'as TARGET' was specified), bind to
  // result to a named TARGET vardecl, inferring its type.
  CValue enterResult =
      getEmitter().emitNamedMethodCall("__enter__", std::move(enterOperands));

  // Create the temporary error decl for any value thrown out of this scope.
  VarDeclOp errDecl = getEmitter().emitVarDecl(
      "__with_error__", UnresolvedType::get(shared.getContext()), loc,
      VarDeclKind::Synthesized);

  // Restore the builder to its current insertion point after parsing.
  llvm::SaveAndRestore builderSaver(builder);
  auto tryOp = TryOp::create(builder, loc, errDecl, /*suppressWarnings=*/true);
  // Stub the 'except' and 'else' regions.
  builder.createBlock(&tryOp.getExceptRegion());

  // If the body of this try can't throw, mark the except region so expressions
  // in it know that.
  if (!inExceptRegion)
    UnreachableOp::create(builder, loc);
  builder.createBlock(&tryOp.getElseRegion());
  TryYieldOp::create(builder, loc);
  builder.createBlock(&tryOp.getTryRegion());

  // Check to see if we have to emit a conditional finally because there is an
  // __exit__ method that accepts an error.  PEP343 states that the general
  // 'with' statement corresponds to:
  //
  //   contextMgr = EXPRESSION
  //   TARGET = contextMgr.__enter__()
  //   exc = True
  //   try:
  //     try:
  //       SUITE
  //     except e:
  //       exc = False
  //       if not contextMgr.__exit__(e):
  //         raise e
  //   finally:
  //     if exc:
  //       contextMgr.__exit__()
  // If the context manager has no __exit__ taking an error, then we know the
  // exit is unconditional.
  Value excVar;
  TryOp nestedTryOp;
  VarDeclOp nestedErrDecl;
  if (inExceptRegion && hasExitMethod) {
    CallOperands exitCallOperands(CallSyntax::kMethodCall, contextExp,
                                  EC_WithExitResult);
    exitCallOperands.addSelf({contextVal, contextExp});
    // We allow any error type for this lookup so we use
    // NameLookupArgWildcardType.
    auto wildcardType = NameLookupArgWildcardType::get(shared.getContext());
    // TODO: We will ultimately pass the Error value in as an MValue, so we
    // could work harder to work with overloads that expect a ref or mut
    // argument.
    exitCallOperands.add({PValue(UnknownAttr::get(wildcardType)), contextExp});

    IREmitter exitEmitter = getEmitter();
    PValue conditionalExit = OverloadSet::lookupAndResolve(
        contextRVType, "__exit__", exitCallOperands, exitEmitter);

    if (conditionalExit) {
      // Insert the flag ahead of our try and initialize it to 'True'.
      OpBuilder::InsertPoint ip = builder.saveInsertionPoint();
      builder.setInsertionPoint(tryOp);

      auto boolType = SIMDType::getScalarBoolType(builder.getContext());
      excVar = getEmitter().emitVarDecl("__with_exc__", boolType, loc,
                                        VarDeclKind::Synthesized);
      RefStoreOp::create(
          builder, loc,
          ParamConstantOp::create(
              builder, loc,
              SIMDAttr::getScalarBool(builder.getContext(), true)),
          excVar);
      builder.restoreInsertionPoint(ip);

      // If the __exit__ method accepts a concrete error type, impose that on
      // the error VarDecl so that throws within the region will conform to it.
      // If it is something generic, then allow the try to resolve it, and the
      // exit call can conform to it.
      ASTType errorType = UnresolvedType::get(shared.getContext());
      auto sigType =
          FnOrFnLiteralTypeGeneratorType::get(conditionalExit.getType());
      assert(sigType.getNumArguments() >= 2 &&
             "expected a receiver and an error");
      auto argType = RefType::stripRefConvention(sigType.getArgument(1),
                                                 sigType.getArgConvention(1));
      // If the method was generic over argument type, then it will get inferred
      // to the wildcard type.  Just leave it as Unresolved if so.
      if (!sugarIsa<NameLookupArgWildcardType>(argType))
        errorType = argType;

      // Generate the nested try. Stub the 'else' and 'finally' regions.
      nestedErrDecl = getEmitter().emitVarDecl("__inner_error__", errorType,
                                               loc, VarDeclKind::Synthesized);
      nestedTryOp =
          TryOp::create(builder, loc, nestedErrDecl, /*suppressWarnings=*/true);
      TryYieldOp::create(builder, loc);
      builder.createBlock(&nestedTryOp.getElseRegion());
      TryYieldOp::create(builder, loc);
      builder.createBlock(&nestedTryOp.getFinallyRegion());
      TryYieldOp::create(builder, loc);

      // Parse the body into the try region.
      builder.createBlock(&nestedTryOp.getTryRegion());
    }
  }

  if (consumeIf(Token::comma)) {
    // We get here if the `with` statement had multiple clauses, like:
    //     with MyClass() as a, MyClass() as b:
    //         ...
    // so recurse to handle them and interpret it as:
    //     with MyClass() as a:
    //         with MyClass() as b:
    //             ...
    // The base case of this recursion call will also handle parsing the body
    // suite for us, so our current call doesn't have to worry about that.
    if (parseSingleWithStmt(curIndent, smLoc, loc))
      return success();
    TryYieldOp::create(builder, loc);
  } else if (consumeIf(Token::colon)) {
    if (parseLocalScopeSuite(curIndent))
      return failure();
    TryYieldOp::create(builder, loc);
  } else {
    auto message = "expected ':' or ',' after 'with' expression";
    auto diagLoc = getTokenLocOrEndOfPreviousLineIfOnNewLine();
    // Report the error.
    auto diag = emitError(diagLoc, message);
    return failure();
  }

  // Now that we emitted the body, we can have inferred the error type. If
  // nothing threw, then infer to Error.  It won't get used and this will avoid
  // possibly confusing diagnostics downstream.
  auto errorVarDecl = nestedErrDecl ? nestedErrDecl : errDecl;
  if (isa<UnresolvedType>(errorVarDecl.getType().getElementType())) {
    if (auto errorType =
            shared.lookupBuiltinType("Error", getParentDecl(), smLoc))
      errorVarDecl.changeElementType(errorType);
  }

  // This emits the call to the 'contextMgr.__exit__()' methods on the
  // context managers in the normal path.  If the type has no __exit__ method,
  // then we extend the result of the __enter__ method with this pattern:
  //
  //   TARGET = contextMgr.__enter__()
  //   try:
  //     SUITE
  //   finally:
  //     lit.ownership.use(TARGET)
  auto emitNormalExitLogic = [&]() {
    // If the target value has no __exit__ method, we need it to be
    // live all the way across the suite, so add an extra use so it isn't
    // destroyed early.
    if (!hasExitMethod) {
      // We don't care about extending PValues if one ever happened.
      if (auto targetBV = getEmitter().emitBValue(
              {enterResult, contextExp}, ExprContext::EC_WithContextMgr)) {
        if (Value ptrOrScalar = enterResult.getMlirValue())
          OwnershipUseOp::create(builder, loc, ptrOrScalar);
      }
      return;
    }

    // The normal exit logic could be the last use of the context manager.  The
    // exit method may take any of "ref", "mut", or "var".  If it is "var", we
    // pass in an RValue for it so it can consume the context manager.
    AnyValue contextVal = MLValue(contextMgrDecl);
    auto exitEmitter = getEmitter();
    CallOperands operands(CallSyntax::kMethodCall, contextExp,
                          EC_WithExitResult);
    operands.addSelf({contextVal, contextExp});
    if (PValue exitMethod = OverloadSet::lookupAndResolve(
            contextRVType, "__exit__", operands, exitEmitter)) {
      // Pass the argument as an RValue so the exit method can consume the
      // value... unless it takes self 'mut'.
      auto signature =
          FnOrFnLiteralTypeGeneratorType::tryGet(exitMethod.getType());
      if (signature.has_value() && !signature->getArgConventions().empty())
        if (signature->getArgConventions()[0] != ArgConvention::Mut) {
          contextVal = MRValue(contextMgrDecl);
        }
    }

    // Ok, emit the call the __exit__.
    (void)getEmitter().emitNamedMethodCall(
        "__exit__",
        CallOperands(CallSyntax::kMethodCall, contextExp, EC_WithExitResult,
                     {{contextVal, contextExp}}));
  };

  // If the body of this try can throw, then the "except" block in it needs to
  // catch the current exception and then re-raise it.
  auto emitExceptBody = [&]() {
    builder.setInsertionPointToStart(&tryOp.getExceptRegion().front());
    if (inExceptRegion) {
      // If nothing in the except region raised, the error vardecl will still be
      // unresolved, just mark the except region as unreachable to silence
      // downstream warnings.
      if (isa<UnresolvedType>(errDecl.getType().getElementType())) {
        UnreachableOp::create(builder, loc);
      } else {
        ExprDest dest(errSlot, EC_RaiseValue);
        // If the contextual caught type is unresolved, then we're the first
        // raise in a try block.  Resolve the error type to whatever we are
        // raising.
        if (isa<UnresolvedType>(errSlot.getRValueType())) {
          auto errorVar = cast<VarDeclOp>(errSlot.getDefiningOp());
          dest = ExprDest(errorVar, EC_RaiseValue);
          getEmitter().checkInferredErrorType(
              errDecl.getType().getElementType(), smLoc);
        }

        getEmitter().emitResult(MRValue(errDecl), contextExp, dest);
        LIT::RaiseOp::create(builder, loc);
        TryYieldOp::create(builder, loc);
      }
    } else {
      // We already emitted the unreachable if this was not reachable.
      assert(isa<UnreachableOp>(builder.getInsertionBlock()->front()));
    }
  };

  // If we're in a non-raising region (or have no __exit__ method), then we have
  // a simple pattern to emit:
  //   contextMgr = EXPRESSION
  //   TARGET = contextMgr.__enter__()
  //   try:
  //     SUITE
  //   finally:
  //     contextMgr.__exit__()
  if (!inExceptRegion || !hasExitMethod) {
    emitExceptBody();
    builder.createBlock(&tryOp.getFinallyRegion());
    emitNormalExitLogic();
    TryYieldOp::create(builder, loc);
    return success();
  }

  // Handle the case when we have a nested try due to an __exit__ method that
  // takes an error.
  if (nestedTryOp) {
    // Set up the except region for the nested try.  Pseudo code:
    //  except(%__inner_error__ : Error) {
    //    %stop_rethrow = contextMgr.__exit__(%__inner_error__);
    //    hlcf.elif %stop_rethrow {
    //      hlcf.yield
    //    } else {
    //      raise %inner_error
    //    }
    builder.createBlock(&nestedTryOp.getExceptRegion());

    // Set the flag to 'False'.
    RefStoreOp::create(
        builder, loc,
        ParamConstantOp::create(
            builder, loc, SIMDAttr::getScalarBool(builder.getContext(), false)),
        excVar);

    // Pass the error value to the __exit__ method.
    // TODO: this isn't using the same convention that Python does.  We support
    // overloading though and this is going to be way better for anything real
    // that wants to implement this. We can support both styles when we need to.
    CallOperands exitOperandList(CallSyntax::kMethodCall, contextExp,
                                 EC_WithExitResult,
                                 {{MLValue(contextMgrDecl), contextExp},
                                  {MBValue(nestedErrDecl), contextExp}});
    auto emitter = getEmitter();
    CValue exitResult =
        emitter.emitNamedMethodCall("__exit__", std::move(exitOperandList));
    RValue exitI1RVal =
        emitter.emitScalarBool({exitResult, contextExp}, EC_WithExitResult);
    SRValue exitI1Val =
        emitter.emitSRValue({exitI1RVal, contextExp}, EC_WithExitResult);
    if (!exitI1Val)
      // Fail, but non-fatal so return success to keep parsing.
      return success();
    // If __exit__ returns false, then re-raise the error.
    Operation *stopRethrowIf = emitter.emitIfThen(
        loc, exitI1Val, TypeRange{},
        [&]() -> LogicalResult {
          // On true, nothing is to be done.
          HLCF::YieldOp::create(*emitter.builder, loc);
          return success();
        },
        [&]() -> LogicalResult {
          // On false, we re-raise the error. Sync the statement builder so
          // getEmitter() emits into this else region.
          builder = *emitter.builder;
          ExprDest dest(MLValue(errDecl), EC_RaiseValue);
          // If the error type is unresolved, resolve it to whatever we
          // propagate.
          if (isa<UnresolvedType>(errDecl.getType().getElementType()))
            dest = ExprDest(errDecl, EC_RaiseValue);
          getEmitter().emitResult(MRValue(nestedErrDecl), contextExp, dest);
          LIT::RaiseOp::create(builder, loc);
          HLCF::YieldOp::create(builder, loc);
          return success();
        });
    if (!stopRethrowIf)
      return success();
    builder.setInsertionPointAfter(stopRethrowIf);
    TryYieldOp::create(builder, loc);
  }

  // Now that we have seen the body of the try, we can have inferred the thrown
  // type, so we can emit the rethrow logic in except {}
  emitExceptBody();

  // Emit the conditional call to __exit__.
  builder.createBlock(&tryOp.getFinallyRegion());
  (void)handleRaisingFinallyRegion(tryOp, smLoc, [&]() -> ParseResult {
    if (!nestedTryOp) {
      emitNormalExitLogic();
      return success();
    }

    // Only call the no-error __exit__ when the suite completed without an
    // exception (exc flag still true).
    Value excFlag = RefLoadOp::create(builder, loc, excVar);
    auto emitter = getEmitter();
    Operation *excIf = emitter.emitIfThen(
        loc, excFlag, TypeRange{},
        [&]() -> LogicalResult {
          builder = *emitter.builder;
          emitNormalExitLogic();
          HLCF::YieldOp::create(builder, loc);
          return success();
        },
        [&]() -> LogicalResult {
          HLCF::YieldOp::create(*emitter.builder, loc);
          return success();
        });
    if (!excIf)
      return failure();
    builder.setInsertionPointAfter(excIf);
    return success();
  });

  TryYieldOp::create(builder, loc);
  return success();
}

ParseResult StmtParser::parseParamIf(Location ifLoc, LexerCursor startCursor,
                                     size_t curIndent) {
  // We will be moving the builder into sub-regions that are created, make sure
  // we end up after it when this is done.
  llvm::SaveAndRestore builderSaver(builder);
  ExprNode *condExp = nullptr;
  if (parseExpression(condExp, curIndent, Precedence::kAssignExpr))
    return failure();

  // Each if/elif conditions could be dynamic or static, use some helpers to
  // generate the right structure.
  ParamIfOp paramIfOp;
  auto parseCondAndTerminateElifCondition = [&](Location loc) -> ParseResult {
    // For a comptime if we emit the condition as a PValue
    // without a builder.
    RValue condRVal = getParamEmitter(EC_ComptimeIfCondition)
                          .emitExprScalarBool(condExp, EC_ComptimeIfCondition);
    if (!condRVal)
      return failure();
    PValue condPVal = condRVal.getIfPValue();
    if (!condPVal)
      return emitError(
                 condExp->getLoc(),
                 "'comptime if' condition must be evaluable at compile-time")
             << condExp->getRange();

    paramIfOp = ParamIfOp::create(builder, loc, condPVal.get());
    return success();
  };

  auto buildBranchAssumption = [&](Location loc,
                                   bool invertCondition) -> ConstraintAttr {
    return LIT::buildBranchAssumption(paramIfOp.getCond(), invertCondition,
                                      loc);
  };

  // Parse a nested suite inside a param-if region. Inserts the branch
  // assumptions before parsing the suite.
  auto parseParamIfRegion =
      [&](ArrayRef<ConstraintAttr> assumptions) -> ParseResult {
    DebugInfo::DIBuilder::ScopeGuard scopeGuard;
    llvm::SaveAndRestore<ASTDecl *> keepDecl(curDeclScope);
    pushChildScope(scopeGuard, keepDecl);
    curDeclScope->insertKnownAssumptions(assumptions);
    return parseSuite(curIndent);
  };

  SmallVector<ConstraintAttr> accumulatedFalseAssumptions;
  Location currentConditionLoc = ifLoc;

  if (parseCondAndTerminateElifCondition(ifLoc) ||
      parseToken(Token::colon, "expected ':' after 'if' expression"))
    return failure();
  builder.createBlock(&paramIfOp.getThenRegion());
  ConstraintAttr currentTrueAssumption =
      buildBranchAssumption(currentConditionLoc, /*invertCondition=*/false);
  if (failed(parseParamIfRegion({currentTrueAssumption})))
    return failure();
  ParamYieldOp::create(builder, ifLoc);

  while (getToken().is(Token::kw_elif) &&
         isTokenInCurrentStatement(curIndent, /*allowSameIndent=*/true)) {
    Location elifLoc = translateLocation(consumeToken(Token::kw_elif).getLoc());
    accumulatedFalseAssumptions.push_back(
        buildBranchAssumption(currentConditionLoc, /*invertCondition=*/true));
    if (parseExpression(condExp, std::nullopt, Precedence::kAssignExpr))
      return failure();

    // Moves emission into "Condition" block if elif.
    builder.createBlock(&cast<ParamIfOp>(paramIfOp).getElseRegion());

    if (parseCondAndTerminateElifCondition(elifLoc) ||
        parseToken(Token::colon, "expected ':' after 'elif' expression"))
      return failure();
    currentConditionLoc = elifLoc;

    ParamYieldOp::create(builder, elifLoc);
    builder.createBlock(&paramIfOp.getThenRegion());
    SmallVector<ConstraintAttr> thenAssumptions(accumulatedFalseAssumptions);
    thenAssumptions.push_back(
        buildBranchAssumption(currentConditionLoc, /*invertCondition=*/false));
    if (failed(parseParamIfRegion(thenAssumptions)))
      return failure();
    ParamYieldOp::create(builder, elifLoc);
  }

  builder.createBlock(&cast<ParamIfOp>(paramIfOp).getElseRegion());
  if (isTokenInCurrentStatement(curIndent, /*allowSameIndent=*/true) &&
      consumeIf(Token::kw_else)) {
    if (parseToken(Token::colon, "expected ':' after else"))
      return failure();
    accumulatedFalseAssumptions.push_back(
        buildBranchAssumption(currentConditionLoc, /*invertCondition=*/true));
    if (failed(parseParamIfRegion(accumulatedFalseAssumptions)))
      return failure();
  }
  ParamYieldOp::create(builder, ifLoc);
  return success();
}

ParseResult StmtParser::parseElif(Location ifLoc, LexerCursor startCursor,
                                  size_t curIndent) {

  struct DeadCodeInfo {
    /// The value of the constant condition.
    bool conditionValue;

    /// The location of the constant condition expression.
    Location location;

    /// Index into elifRegions of the condition region, or ~0u for the first
    /// (operand) condition.
    unsigned elifCondIndex;
  };

  // We will be moving the builder into sub-regions that are created, make sure
  // we end up after it when this is done.
  llvm::SaveAndRestore builderSaver(builder);

  // Unreachable-arm metadata collected while emitting conditions; processed
  // after the full elif is built so we can mark regions dead.
  SmallVector<DeadCodeInfo> ifOpsWithDeadCode;

  // Emit a condition at the current insertion point. The first `if` condition
  // is emitted before the elif; later `elif` conditions stay in their regions.
  auto emitConditionExpr = [&](unsigned elifCondIndex) -> FailureOr<Value> {
    auto emitter = getEmitter();

    ExprNode *condExp = nullptr;
    if (parseExpression(condExp, curIndent, Precedence::kAssignExpr))
      return failure();

    RValue condI1RVal = emitter.emitExprScalarBool(condExp, EC_BoolCondition);
    if (!condI1RVal)
      return failure();
    if (PValue condI1PVal = condI1RVal.getIfPValue();
        auto asBoolAttr = sugarDynCastIfPresent<SIMDAttr>(condI1PVal.get())) {
      ifOpsWithDeadCode.push_back({asBoolAttr.getAsBool(),
                                   condExp->getLocation(emitter),
                                   elifCondIndex});
    }
    SRValue condRVal =
        emitter.emitSRValue({condI1RVal, condExp}, EC_BoolCondition);
    if (!condRVal)
      return failure();
    return Value(condRVal);
  };

  // First condition is evaluated in the parent and becomes the elif operand.
  FailureOr<Value> firstCond = emitConditionExpr(/*elifCondIndex=*/~0u);
  if (failed(firstCond) ||
      parseToken(Token::colon, "expected ':' after 'if' expression"))
    return failure();

  HLCF::ElifOp elifOp =
      HLCF::ElifOp::create(builder, ifLoc, TypeRange(), *firstCond);
  elifOp.getThenRegion().emplaceBlock();
  elifOp.getElseRegion().emplaceBlock();

  auto appendElifRegionPair = [&]() {
    builder.setInsertionPoint(elifOp);
    IRRewriter rewriter{builder};
    HLCF::ElifOp replacement = HLCF::ElifOp::create(
        builder, elifOp.getLoc(), elifOp->getResultTypes(), elifOp.getCond(),
        elifOp.getElifRegions().size() + 2);

    replacement.getThenRegion().takeBody(elifOp.getThenRegion());
    replacement.getElseRegion().takeBody(elifOp.getElseRegion());
    for (auto [index, source] : llvm::enumerate(elifOp.getElifRegions()))
      replacement.getElifRegions()[index].takeBody(source);

    Region &lastConditionRegion =
        replacement.getElifRegions()[replacement.getElifRegions().size() - 2];
    Region &lastThenRegion = replacement.getElifRegions().back();
    lastConditionRegion.emplaceBlock();
    lastThenRegion.emplaceBlock();

    rewriter.replaceOp(elifOp, replacement);
    elifOp = replacement;
  };

  // Parse Then region.
  builder.setInsertionPointToStart(&elifOp.getThenRegion().front());
  if (failed(parseLocalScopeSuite(curIndent)))
    return failure();
  HLCF::YieldOp::create(builder, ifLoc);

  // Parse Elif chain if it exists. Subsequent conditions stay in their regions
  // so they are only evaluated when earlier arms fail.
  while (getToken().is(Token::kw_elif) &&
         isTokenInCurrentStatement(curIndent, /*allowSameIndent=*/true)) {
    Location elifLoc = translateLocation(consumeToken(Token::kw_elif).getLoc());
    appendElifRegionPair();

    unsigned condRegionIndex = elifOp.getElifRegions().size() - 2;
    builder.setInsertionPointToStart(
        &elifOp.getElifRegions()[condRegionIndex].front());
    FailureOr<Value> elifCond = emitConditionExpr(condRegionIndex);
    if (failed(elifCond) ||
        parseToken(Token::colon, "expected ':' after 'elif' expression"))
      return failure();
    HLCF::ElifYieldOp::create(builder, elifLoc, *elifCond,
                              /*no extra values*/ ValueRange());

    // Parse Then region.
    builder.setInsertionPointToStart(&elifOp.getElifRegions().back().front());
    if (failed(parseLocalScopeSuite(curIndent)))
      return failure();
    HLCF::YieldOp::create(builder, elifLoc);
  }

  builder.setInsertionPointToStart(&elifOp.getElseRegion().front());
  if (isTokenInCurrentStatement(curIndent, /*allowSameIndent=*/true) &&
      consumeIf(Token::kw_else)) {
    if (parseToken(Token::colon, "expected ':' after else"))
      return failure();
    if (failed(parseLocalScopeSuite(curIndent)))
      return failure();
  }
  HLCF::YieldOp::create(builder, ifLoc);

  // Process dead code.  Go backward to avoid needing to erase an already erased
  // region.
  if (!ifOpsWithDeadCode.empty()) {
    for (auto [condition, condExprLoc, elifCondIndex] :
         llvm::reverse(ifOpsWithDeadCode)) {
      shared.emitWarning(condExprLoc)
          << "'if' condition always evaluates to '"
          << (condition ? "True" : "False")
          << (condition ? "'; 'else' branch is unreachable"
                        : "'; 'if' branch is unreachable");
      if (elifCondIndex == ~0u) {
        // First (operand) condition.
        if (condition) {
          markRegionUnreachable(&elifOp.getElseRegion(), ifLoc);
          for (auto &region : elifOp.getElifRegions())
            markRegionUnreachable(&region, ifLoc);
        } else {
          markRegionUnreachable(&elifOp.getThenRegion(), ifLoc);
        }
      } else if (condition) {
        // Additional arm is true: else and later pairs are unreachable.
        markRegionUnreachable(&elifOp.getElseRegion(), ifLoc);
        for (auto &region : elifOp.getElifRegions().slice(elifCondIndex + 2))
          markRegionUnreachable(&region, ifLoc);
      } else {
        // Additional arm is false: its then is unreachable.
        markRegionUnreachable(&elifOp.getElifRegions()[elifCondIndex + 1],
                              ifLoc);
      }
    }
  }

  return success();
}

/// Validates that an import statement appears at a permitted scope: either
/// directly at module scope (FileModuleOp) or at function scope (FnOp).
/// Within a function, comptime control-flow ops (ParamIfOp, ParamForOp) are
/// transparent — the check walks through them. Any other intervening op
/// (e.g. HLCF::IfOp, LIT::LoopOp) indicates runtime control flow and is
/// rejected. Struct, trait, and extension bodies are also rejected.
ParseResult StmtParser::checkImportScope(SMLoc kwLoc) {
  Operation *parent = getParentDecl().getIfOperation();
  // Module scope is always valid.
  if (isa_and_nonnull<FileModuleOp>(parent))
    return success();
  // Within a function, walk up the region chain from the current insertion
  // point to the FnOp. Comptime control-flow ops (ParamIfOp, ParamForOp) are
  // transparent — we continue walking through them. Any other op in between
  // (HLCF::IfOp, LIT::LoopOp, etc.) is runtime control flow and the import
  // is rejected.
  if (isa_and_nonnull<FnOp>(parent)) {
    Block *block = builder.getInsertionBlock();
    while (block) {
      Operation *op = block->getParentOp();
      if (!op)
        break;
      if (isa<FnOp>(op))
        return success();
      if (isa<ParamIfOp, ParamForOp>(op)) {
        block = op->getBlock();
        continue;
      }
      break; // runtime control-flow op
    }
  }
  emitError(kwLoc) << "'import' statements must be at module or function "
                      "scope; move this to a valid location";
  return failure();
}

// TODO: Single-source this validation with Decorators::handleStable() in
// DeclResolution.cpp. The logic is split because import decorator processing
// must happen eagerly at parse time.

/// Validate decorators written before a 'from ... import' statement.
///
/// Re-lexes from startCursor (which points before the leading '@') so that
/// errors are emitted eagerly even when the import is never referenced
/// (resolveSignature is lazy).
///
/// Returns the decorators that were present.
static FromImportDecorators parseFromImportDecorators(SharedState &shared,
                                                      LexerCursor startCursor,
                                                      size_t stmtIndent) {
  FromImportDecorators result;
  Lexer decorLexer(shared.diags, startCursor);
  ParserBase decorParser(shared, decorLexer);
  for (auto [expr, cursor] : decorParser.parseDecorators((ssize_t)stmtIndent)) {
    if (auto *callNode = dyn_cast<CallNode>(expr)) {
      auto *callee = dyn_cast<DeclRefNode>(callNode->callee);
      if (callee && callee->spelling == "stable") {
        if (callNode->operands.size() == 1 &&
            callNode->operands.front().isKeyword() &&
            callNode->operands.front().name == "recursive") {
          auto *boolNode =
              dyn_cast<BoolLiteralNode>(callNode->operands.front().expr);
          if (boolNode && boolNode->value) {
            result.hasStableOverride = true;
            continue; // Valid decorator — proceed.
          }
          shared.emitError(callNode->operands.front().expr->getLoc(),
                           "'recursive' argument to @stable must be True");
          continue;
        }
        shared.emitError(
            expr->getLoc(),
            "@stable on import requires 'recursive=True' argument");
        continue;
      }
    }
    if (auto *declRef = dyn_cast<DeclRefNode>(expr)) {
      if (declRef->spelling == "stable") {
        shared.emitError(
            expr->getLoc(),
            "@stable on import requires 'recursive=True' argument");
        continue;
      }
      if (declRef->spelling == "__doc_inline") {
        result.docInline = true;
        continue;
      }
    }
    shared.emitError(expr->getLoc(),
                     "'from' statement supports only the @stable and "
                     "@__doc_inline decorators; remove the decorator");
  }
  return result;
}

/// import_stmt     ::=  "from" relative_module "import" identifier
///                        ["as" identifier] ("," identifier ["as" identifier])*
///                      | "from" relative_module "import" "(" identifier
///                        ["as" identifier] ("," identifier ["as" identifier])*
///                        [","] ")"
///                      | "from" relative_module "import" "*"
/// module          ::=  (identifier ".")* identifier
/// relative_module ::=  "."* module | "."+
ParseResult StmtParser::parseFromImportStmt(FromImportDecorators decorators) {
  SMLoc kwLoc = getToken().getLoc();
  consumeToken(Token::kw_from);

  if (failed(checkImportScope(kwLoc)))
    return failure();

  SMLoc importLoc = getToken().getLoc();
  SharedState::ImportPath modulePath;
  if (parseImportModuleName(modulePath, /*allowRelativeImport=*/true))
    return failure();
  auto nextTok = getToken();
  if (parseToken(Token::kw_import, "expected 'import' after module name") ||
      rejectTokenAtStartOfLine(nextTok, "'import' statement"))
    return failure();

  auto moduleAttr = modulePath.toAttr(getContext());

  // Check for a wildcard import.
  nextTok = getToken();
  if (consumeIf(Token::star)) {
    if (rejectTokenAtStartOfLine(nextTok, "wildcard import"))
      return failure();
    if (decorators.hasStableOverride)
      emitError(importLoc, "@stable(recursive=True) is not supported on "
                           "wildcard imports");
    if (decorators.docInline)
      emitError(importLoc,
                "@__doc_inline is not supported on wildcard imports");
    LIT::UnresolvedWildcardImportOp::create(
        builder, translateLocation(importLoc), moduleAttr);
    getParentDecl().addUnresolvedWildcardImport(
        UnresolvedWildcardImport{moduleAttr, importLoc});
    return success();
  }

  // A functor used to signal to any parser listener that we're importing a decl
  // from the module. If we do emit any notifications, keep track of the
  // currently resolved parent module/package so that the listener can have
  // context for the import.
  ASTDecl *currentResolvedModule = nullptr;
  auto notifyListenerOfImport = [&]() {
    if (!shared.parserListener)
      return;
    SMLoc loc = getToken().getLoc();
    shared.notifyListenerOnMemberLookup(loc, [&]() -> ASTDecl & {
      // Resolve the module if we haven't yet.
      if (!currentResolvedModule) {
        ASTDecl *curModuleDecl = curDeclScope;
        while (curModuleDecl &&
               !isa_and_nonnull<FileModuleOp>(curModuleDecl->getIfOperation()))
          curModuleDecl = curModuleDecl->getParentDecl();

        currentResolvedModule = &shared.importModule(
            modulePath,
            curModuleDecl
                ? curModuleDecl->getIfOperation()->getParentOfType<PackageOp>()
                : PackageOp(),
            importLoc);
      }
      return *currentResolvedModule;
    });
  };

  // Parse the set of constructs to import.
  nextTok = getToken();
  bool isTupleImport = consumeIf(Token::l_paren);

  if (isTupleImport &&
      rejectTokenAtStartOfLine(nextTok, "beginning of tuple import"))
    return failure();

  do {
    // Parse the next construct to import.
    SMLoc importSourceNameLoc = getToken().getLoc();
    StringRef importSourceName = getTokenSpelling();
    nextTok = getToken();
    bool missingIdentifier =
        failed(parseIdentifier("expected construct name to import"));
    notifyListenerOfImport();

    // If there was no identifier, then we're done.
    if (missingIdentifier)
      return failure();
    if (!isTupleImport &&
        rejectTokenAtStartOfLine(nextTok, "construct name to import"))
      return failure();
    StringRef importDestName = importSourceName;
    SMLoc importDestLoc = importSourceNameLoc;
    nextTok = getToken();
    if (consumeIf(Token::kw_as)) {
      if (rejectTokenAtStartOfLine(nextTok, "'as' keyword"))
        return failure();
      importDestName = getTokenSpelling();
      importDestLoc = getToken().getLoc();
      nextTok = getToken();
      if (parseIdentifier("expected name to import '" + importSourceName +
                          "' as"))
        return failure();
      if (rejectTokenAtStartOfLine(nextTok, "bound import name"))
        return failure();
    }

    // Create an unresolved decl for this import.
    StringAttr importDestNameAttr = builder.getStringAttr(importDestName);
    auto importDecl = LIT::UnresolvedImportOp::create(
        builder, translateLocation(importLoc), moduleAttr, importDestNameAttr,
        builder.getStringAttr(importSourceName),
        translateLocation(importDestLoc),
        translateLocation(importSourceNameLoc));
    getDeclResolver().addDecl(importDecl, importLoc, importDestNameAttr,
                              curDeclScope, getLexer().getCursor(),
                              getLexer().getCursor(), /*indentation=*/-1);

    if (decorators.hasStableOverride) {
      if (importSourceName != importDestName) {
        // Renamed imports (from mod import X as Y) are unsupported: tracking
        // 'Y' would miss member accesses if code uses 'X', and vice versa.
        shared.emitError(importDestLoc,
                         "@stable(recursive=True) is not supported on renamed "
                         "imports ('import ... as ...')");
      } else {
        // Record the import name in the parent scope for use-site warning
        // suppression. We do this here rather than in resolveSignature because
        // UnresolvedImportOp does not carry the @stable decorator attribute.
        curDeclScope->addRecursivelyStableName(importDestNameAttr);
      }
    }

    if (decorators.docInline) {
      if (importSourceName != importDestName) {
        // Doc generation documents the target's own declaration, which carries
        // the source name, so a rename would publish the wrong one.
        shared.emitError(importDestLoc,
                         "@__doc_inline is not supported on renamed imports "
                         "('import ... as ...')");
      } else {
        curDeclScope->addDocInlineName(importDestNameAttr);
      }
    }

    // Check for more elements to import.
    nextTok = getToken();
    if (!consumeIf(Token::comma))
      break;
    if (rejectTokenAtStartOfLine(nextTok, "comma"))
      return failure();
    // For tuple imports, there may optionally be a trailing comma at the end of
    // the list.
    if (isTupleImport && getToken().is(Token::r_paren))
      break;
  } while (true);

  // Check for the end of the tuple import.
  if (isTupleImport &&
      parseToken(Token::r_paren, "expected ')' after import list"))
    return failure();
  return success();
}

/// Build a chain of nested ImportOps for the module path
/// \p modulePath (e.g. `a.b.c`) into \p dest's scope, reusing existing
/// ImportOps at shared prefixes. This is the resolved form of a plain
/// `import a.b.c`; it is built eagerly at parse time so the bound name `a`
/// (and `a.b`, `a.b.c`) are resolvable before any reference to them.
static void buildImportChain(ParserBase &p, ASTDecl &dest, OpBuilder builder,
                             const SharedState::ImportPath &modulePath,
                             mlir::Location loc) {
  assert(modulePath.relativeLevel == 0 && "import chains are absolute");
  ASTDecl *scope = &dest;
  ArrayRef<StringRef> components = modulePath.components;
  for (unsigned i = 0, e = components.size(); i != e; ++i) {
    auto segName = StringAttr::get(p.getContext(), components[i]);

    // Reuse the ImportOp already bound at this scope; otherwise create a new
    // one.
    ASTDecl *node = nullptr;
    for (ASTDecl *d : scope->lookupInCurrentScope(segName)) {
      if (isa_and_nonnull<ImportOp>(d->getIfOperation())) {
        node = d;
        break;
      }
    }
    if (!node) {
      node = &p.getDeclResolver().createImportOp(
          *scope, builder, segName,
          ImportPathAttr::get(p.getContext(), /*relativeLevel=*/0,
                              components.take_front(i + 1)),
          loc);
    }

    // Descend so the next segment is gated as a child of this node.
    scope = node;
    builder = scope->getDeclEndBuilder();
  }

  p.shared.notifyListenerOnModuleImport(*scope, modulePath,
                                        p.shared.diags.convertLocToSMLoc(loc));
}

/// import_stmt ::=  "import" module ["as" identifier]
///                  ("," module ["as" identifier])*
/// module      ::=  (identifier ".")* identifier
ParseResult StmtParser::parseImportStmt() {
  SMLoc kwLoc = getToken().getLoc();
  consumeToken(Token::kw_import);

  if (failed(checkImportScope(kwLoc)))
    return failure();

  // Parse the next module to import.
  auto nextTok = getToken();
  do {
    if (nextTok.is(Token::comma) && rejectTokenAtStartOfLine(nextTok, "comma"))
      return failure();
    SMLoc importLoc = getToken().getLoc();
    SharedState::ImportPath modulePath;
    if (parseImportModuleName(modulePath, /*allowRelativeImport=*/false))
      return failure();

    auto moduleAttr = modulePath.toAttr(getContext());

    // Check for a name binding. 'import a.b.c as z' is a *leaf binding*: the
    // name 'z' binds the resolved (leaf) module directly, rather than 'import
    // a.b.c' which binds each module along the dotted path.
    std::optional<StringRef> boundModuleName;
    SMLoc boundNameLoc;
    nextTok = getToken();
    if (consumeIf(Token::kw_as)) {
      if (rejectTokenAtStartOfLine(nextTok, "'as' keyword"))
        return failure();
      boundModuleName = getTokenSpelling();
      boundNameLoc = getToken().getLoc();
      nextTok = getToken();
      if (parseIdentifier("expected name to bind import"))
        return failure();
      if (rejectTokenAtStartOfLine(nextTok, "bound import name"))
        return failure();
    }

    // Absolute `import`s are built into nested ImportOps at parse time, with no
    // UnresolvedImportOp placeholder. Resolve the target module first so a
    // missing module is reported eagerly (this locates the module file; it does
    // not parse its body).
    ASTDecl &module =
        shared.importModule(modulePath, /*currentPackage=*/nullptr, importLoc);

    if (!boundModuleName.has_value()) {
      // 'import a.b.c' binds the chain 'a' -> 'a.b' -> 'a.b.c' under the
      // first segment 'a'.
      buildImportChain(*this, *curDeclScope, builder, modulePath,
                       translateLocation(importLoc));
      continue;
    }

    // 'import a.b.c as z' binds a single gate 'z' -> 'a.b.c'; the names 'a'
    // and 'a.b' are not bound.
    getDeclResolver().createImportOp(*curDeclScope, builder,
                                     builder.getStringAttr(*boundModuleName),
                                     moduleAttr, translateLocation(importLoc));
    shared.notifyListenerOnModuleImport(module, modulePath, importLoc);
    // Index the bound alias name (`z` in `import a.b.c as z`) as a
    // reference to the imported module so hover/semantic tokens resolve it.
    if (boundNameLoc.isValid())
      shared.notifyListenerOnRef(&module, *boundModuleName, boundNameLoc);
    nextTok = getToken();
  } while (consumeIf(Token::comma));

  return success();
}

/// Parse a module name for use in an import statement.
/// module          ::=  (identifier ".")* identifier
/// relative_module ::=  "."* module | "."+
///
/// Relative forms are only allowed if allowRelativeImport is 'true'.
ParseResult
StmtParser::parseImportModuleName(SharedState::ImportPath &parsedName,
                                  bool allowRelativeImport) {
  // A functor used to signal to any parser listener that we're importing a
  // module.
  auto notifyListenerOfImport = [&]() {
    if (!shared.parserListener)
      return;
    SMLoc loc = getToken().getLoc();

    // If there isn't a module name, this is a top-level import.
    if (parsedName.components.empty() && parsedName.relativeLevel == 0)
      return shared.notifyListenerOnImport(loc);

    // Otherwise, this is importing from within a package.
    shared.notifyListenerOnImport(loc, [&]() -> ASTDecl & {
      auto curOp = curDeclScope->getIfOperation()->getParentOfType<PackageOp>();
      return shared.importModule(parsedName, curOp, loc);
    });
  };

  if (rejectTokenAtStartOfLine("module path"))
    return failure();

  // Cache whether we're parsing a relative import so we can emit a good
  // diagnostic if we're in a context where they're not permitted.
  SMLoc relativeErrorLoc = getToken().getLoc();
  bool parsedRelativeImport =
      getToken().is(Token::dot) || getToken().is(Token::dot_dot_dot);

  // Parse the relative '.' indicators that resolve to a parent package.
  while (true) {
    auto nextTok = getToken();
    if (consumeIf(Token::dot))
      parsedName.relativeLevel += 1;
    else if (consumeIf(Token::dot_dot_dot))
      parsedName.relativeLevel += 3;
    else
      break;
    if (rejectTokenAtStartOfLine(nextTok, "module path"))
      return failure();
  }

  // If we have a non-relative module name, or we require one, try to parse it.
  if (parsedName.relativeLevel == 0 || getToken().isIdentifier()) {
    // Parse the first module name.
    StringRef rootModuleName = getTokenSpelling();
    bool missingIdentifier = failed(parseIdentifier(
        "expected module name", /*loc=*/nullptr, /*forbidStartOfLine=*/true));
    notifyListenerOfImport();

    // If there was no identifier, then we're done.
    if (missingIdentifier)
      return failure();

    parsedName.components.push_back(rootModuleName);

    // Parse nested module names.
    auto nextTok = getToken();
    while (consumeIf(Token::dot)) {
      notifyListenerOfImport();

      if (rejectTokenAtStartOfLine(nextTok, "module path"))
        return failure();
      nextTok = getToken();

      parsedName.components.push_back(getTokenSpelling());
      if (parseIdentifier("expected module name"))
        return failure();

      if (rejectTokenAtStartOfLine(nextTok, "module path"))
        return failure();
      nextTok = getToken();
    }
  } else {
    notifyListenerOfImport();
  }

  if (parsedRelativeImport && !allowRelativeImport) {
    auto diag = emitError(relativeErrorLoc)
                << "relative imports must use 'from'";
    // If the user has provided a module name, emit a more helpful diagnostic
    // pointing them towards "from [.]+ import foo". If they've just written
    // "import [.]+" we can't point them towards valid syntax from here.
    auto moduleName = llvm::join(parsedName.components, ".");
    if (!moduleName.empty()) {
      diag << "; did you mean 'from "
           << std::string(parsedName.relativeLevel, '.') << " import "
           << moduleName << "'?";
    }
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Definition statements
//===----------------------------------------------------------------------===//

ParseResult StmtParser::parseDefFnStmt(LexerCursor startCursor,
                                       size_t curIndent) {
  if (consumeIf(Token::kw_async) && rejectTokenAtStartOfLine("'def' keyword"))
    return failure();
  consumeToken(); // Consume either 'def' or 'fn'.

  SMLoc loc;
  StringAttr baseName;
  // Reject a hard keyword as a free-function name
  bool nameIsKeyword = getToken().isKeyword();
  if (parseIdentifier(baseName, "expected function name", &loc,
                      /*forbidStartOfLine=*/true,
                      /*allowKeyword=*/true))
    return failure();

  // Canonicalize deprecated special-function spellings (e.g. '__del__') to
  // their canonical form as early as possible.
  if (StringRef canonical =
          SpecialFunctionInfo::getCanonicalSpelling(baseName.getValue());
      canonical != baseName.getValue())
    baseName = StringAttr::get(getContext(), canonical);

  if (nameIsKeyword && !isInTypeBody()) {
    emitError(loc) << "'" << baseName.getValue()
                   << "' cannot be used as a function name in this context";
  }

  // The parameter/argument list must begin on the same line as the function
  // name. Otherwise the decl-extent scan below (skipUntilIndentation) stops at
  // the start-of-line bracket and truncates the declaration, which surfaces as
  // a confusing internal error during signature resolution.
  if (getToken().is(Token::l_square) &&
      rejectTokenAtStartOfLine("parameter list"))
    return failure();
  if (getToken().is(Token::l_paren) &&
      rejectTokenAtStartOfLine("argument list"))
    return failure();

  // Create a op function with an empty signature so we have an IR construct to
  // work with.

  // Before resolution, we treat the function as having type ()->Error,
  // because parse or other errors forming the signature won't update the
  // representation.  This makes sure that the error case doesn't break
  // invariants (that functions always have a single result).
  MLIRContext *ctx = builder.getContext();
  auto errorType = builder.getType<TypeCheckErrorType>();
  auto valueTypes = FunctionType::get(ctx, ArrayRef<Type>(), {errorType});
  size_t numInputs = valueTypes.getNumInputs();
  SmallVector<PogMetadataAttr> argPogs(
      numInputs,
      PogMetadataAttr::get(StringAttr::get(ctx), PassingKind::PosOnly));
  auto pogList = PogListAttr::get(ctx, argPogs);
  auto metadata = FnMetaOriginDataAttr::get(ctx, /*numImplicitOriginDecls=*/0,
                                            OriginSetAttr::get(ctx, {}),
                                            /*isNestedOriginsReadOnly=*/false,
                                            /*definesInteriorOrigins=*/false);
  auto funcSig = FuncType::get(valueTypes, metadata, pogList);
  FnTypeGeneratorType signatureType = GeneratorType::get(
      /*inputParamTypes=*/{}, funcSig, PogListAttr::get(ctx));

  // Chain to the 'build' method below.
  auto emptyStr = StringAttr::get(ctx, "");
  auto fnOp = FnOp::create(builder, translateLocation(loc), emptyStr, emptyStr,
                           signatureType);

  // NOTE: We set an attribute named 'sym_namex' here instead of setting
  // 'sym_name' because we don't /know/ the symbol name on construction and need
  // to set it during signature resolution phase of the parser.
  //
  // Unfortunately, we cannot set it to null because that causes the SymbolTable
  // logic to be extremely cranky and breaks other MLIR invariants.
  //
  // We also cannot completely omit the symbol, because ODS is doing some clever
  // stuff to speed up attribute lookup.  That clever stuff requires that a slot
  // is filled in the attr dict, so we set this thing and remove it when the
  // real name is set.
  fnOp->removeAttr("sym_name");
  fnOp->setAttr("sym_namex", emptyStr /*StringArrayAttr::get(ctx, {})*/);
  fnOp.setSourceNameAttr(baseName);

  // Mark this function with an attribute if it's trait method with a non-empty
  // body.
  if (isa_and_nonnull<TraitDeclOp>(curDeclScope->getIfOperation()))
    maybeMarkDefaultedTraitMethod(fnOp);

  // Skip the body of this definition: go to a token at the start of the next
  // line at the same indent level (or less) as the current definition.
  skipUntilIndentation(curIndent);
  ASTDecl &funcDecl =
      getDeclResolver().addDecl(fnOp, loc, baseName, curDeclScope, startCursor,
                                getLexer().getCursor(), curIndent);

  // Add an EndFnOp to the end of the body. This makes the function able to
  // verify clean, even if we don't body or signature resolve it.  We may end up
  // removing this when resolving the body.
  auto builder = OpBuilder::atBlockEnd(fnOp.getBody());
  EndFnOp::create(builder, fnOp.getLoc(), /*unresolved=*/true);

  // If this is a nested function, parse its body right now so captures can be
  // resolved correctly.
  if (curDeclScope->getNearestDeclOfType<FnOp>())
    (void)getDeclResolver().resolveBody(funcDecl, loc);
  return success();
}

void StmtParser::maybeMarkDefaultedTraitMethod(FnOp fnOp) {
  // Save the current lexer state so we can restore it after inspection.
  LexerCursor savedCursor = getLexer().getCursor();

  // Skip tokens that compose a function signature. This is used as a helper
  // for determining whether a trait method provides an implementation or not.
  //
  // The actual implementation is exceedingly straightforward and will just
  // consume tokens up to and including the first ':' token as long as it's not
  // nested inside of any parens or square brackets.
  //
  // While this does technically allow syntactically invalid forms like:
  // def foo()[]:, def []()foo:, def foo[](): and others we don't care for the
  // purposes of marking a trait method as defaulted or not since later parsing
  // of the signature will result in a parser failure anyways and the simple
  // logic that is provided is capable of handling valid syntactic forms.
  auto skipSignature = [&]() {
    unsigned parenDepth = 0;
    unsigned squareDepth = 0;
    while (true) {
      switch (getToken().getKind()) {
      case Token::eof:
        return; // Unterminated signature – bail out gracefully.
      case Token::l_paren:
        ++parenDepth;
        consumeToken();
        break;
      case Token::r_paren:
        if (parenDepth)
          --parenDepth;
        consumeToken();
        break;
      case Token::l_square:
        ++squareDepth;
        consumeToken();
        break;
      case Token::r_square:
        if (squareDepth)
          --squareDepth;
        consumeToken();
        break;
      case Token::colon:
        if (parenDepth == 0 && squareDepth == 0) {
          consumeToken(Token::colon); // Eat the terminating ':'
          return;                     // Positioned at first token of the body.
        }
        consumeToken();
        break;
      default:
        consumeToken();
        break;
      }
    }
  };

  skipSignature();

  // Consume the doc string if present.
  StringRef docString;
  parseDocString(docString);

  // Mark the function as defaulted unless its body is explicitly empty.
  if (!getToken().is(Token::dot_dot_dot))
    fnOp.setDefaultedTraitFn(true);

  // Restore lexer state so normal parsing can continue unharmed.
  savedCursor.restore(getLexer());
}

/// var_decl_stmt ::= "var" identifier ":" expression ["=" expression]
///                 | "var" identifier "=" expression
ParseResult StmtParser::parseVarStmt(LexerCursor startCursor,
                                     size_t stmtIndent) {
  assert(!isa_and_nonnull<FnOp>(getParentDecl().getIfOperation()) &&
         "var decls in functions are processed as expression statements");

  // Global var decls are allowed to have decorators, but nothing else.
  bool hasDecorators = startCursor != getLexer().getCursor();
  auto rejectDecorator = [&, declTok = getToken()]() {
    if (!hasDecorators)
      return;
    emitError(declTok.getLoc())
        << "'" << declTok.getSpelling()
        << "' statement does not support decorators; remove the decorator";
  };

  auto smLoc = consumeToken().getLoc();
  auto loc = translateLocation(smLoc);
  SMLoc identifierLoc;
  StringAttr name;
  if (parseIdentifier(name, "expected name for 'var' declaration",
                      &identifierLoc, /*forbidStartOfLine=*/true))
    return failure();

  if (isa_and_nonnull<TraitDeclOp>(getParentDecl().getIfOperation())) {
    rejectDecorator();
    emitError(loc, "traits do not support 'var' fields; use 'comptime' to "
                   "declare associated types");
    skipUntilIndentation(stmtIndent, /*stopOnSemicolon=*/true);
    return success();
  }

  auto unresolvedType = UnresolvedType::get(getContext());
  // If we're in a struct, then this is a field declaration.
  Operation *declOp;
  if (isa_and_nonnull<StructDeclOp>(getParentDecl().getIfOperation())) {
    declOp = StructFieldOp::create(builder, loc, name, unresolvedType);

    // Skip the body of this definition: go to a token the starts a line at the
    // same indent level (or less) as the current definition.
    skipUntilIndentation(stmtIndent, /*stopOnSemicolon=*/true);
  } else {
    emitError(loc, "global variables are not supported; move this into "
                   "a function body or use 'comptime' to declare a constant");
    skipUntilIndentation(stmtIndent, /*stopOnSemicolon=*/true);
    return success();
  }

  // Remember that we parsed this declaration so we can finish type checking it
  // when it gets referenced.
  // If the declaration is in a function body, we delay adding it until after
  // resolving the RHS, so that the RHS can reference any identifiers that the
  // decl is shadowing.
  ASTDecl &decl = getDeclResolver().createUnlistedDecl(
      declOp, smLoc, curDeclScope, startCursor, getLexer().getCursor(),
      stmtIndent);
  getDeclResolver().attachDeclToParentNameTable(&decl, name);

  auto varOp = dyn_cast_or_null<VarDeclOp>(decl.getIfOperation());
  if (!varOp) {
    // Parse docstrings for struct fields here.
    parseDocString(decl);
    return success();
  }

  // Local variable declarations inside functions are lexically resolved, so
  // fully resolve the decl now. If an error occurs, skip the declaration and
  // keep parsing to emit as many diagnostics as possible.
  auto declParseError = [&] {
    decl.setErroneous();
    skipUntilIndentation(stmtIndent, /*stopOnSemicolon=*/true);
    return success();
  };

  // Parse the type if present.
  ASTType parsedType;
  IREmitter emitter = getEmitter();
  if (consumeIf(Token::colon)) {
    ExprNode *typeExpr = nullptr;
    if (parseExpression(typeExpr, stmtIndent))
      return declParseError();
    parsedType = emitter.emitExprType(typeExpr);
    if (!parsedType)
      return declParseError();
  }

  // Parse the initializer if present.
  ExprNode *initExpr = nullptr;
  if (consumeIf(Token::equal)) {
    if (parseVarInitExpression(initExpr, stmtIndent))
      return declParseError();
  }

  // Now that parsing succeeded, we do IR emission and semantic processing.

  // Handle the initializer if present.
  if (initExpr) {
    // If we have a type, then emit directly into the LValue.  Otherwise emit
    // into the varOp to infer its type.
    ExprDest dest(EC_VarInit);
    ExprContext exprContext = EC_VarInit;
    if (parsedType) {
      varOp.changeElementType(parsedType);
      dest = ExprDest(MLValue(varOp), exprContext);
    } else {
      // If we don't, we emit into the varOp itself, because this will infer the
      // type of the varOp from the initializer expression.
      dest = ExprDest(varOp, exprContext);
    }

    if (!emitter.emitExpr(initExpr, dest))
      return declParseError();

    assert(!isa<UnresolvedType>(varOp.getType().getElementType()) &&
           "RValue emission should have inferred var type");

  } else if (parsedType) {
    varOp.changeElementType(parsedType);
  } else {
    // If there was neither a type or initializer, reject the var.
    emitError(varOp.getLoc(),
              "declaration must have either a type or an initializer");
    return declParseError();
  }

  // Now mark the decl as fully resolved.
  decl.resolvedness = DeclResolvedness::body;

  shared.notifyListenerOnVariableDecl(decl, identifierLoc);
  return success();
}

// Parse the targets for an alias declaration stmt, if this is a single target
// alias declaration (either a parametric alias or a single identifier), return
// the name for the target, else return a ExprNode* for the target expression.
static ParseResult
parseAliasDeclTargetsExpr(ParserBase &p,
                          SmartVariant<StringRef, ExprNode *> &result,
                          size_t stmtIndent) {

  if (p.getLexer().getToken().isIdentifier()) {
    if (p.rejectTokenAtStartOfLine("identifier"))
      return failure();
    LexerCursor cursor(p.getLexer());
    result = p.consumeIdentifier().getSpelling();
    // One of the cases below:
    // 1. alias P[a: Int, //, ..]
    // 2. alias P = ...
    // 3. alias P : Bool ...
    // 4. alias P where ... = ...  ('where' is a soft keyword)
    Token nextTok = p.getLexer().getToken();
    if (nextTok.is(Token::l_square) || nextTok.is(Token::equal) ||
        nextTok.is(Token::colon) ||
        (nextTok.isIdentifier() && nextTok.getSpelling() == "where"))
      return success();

    // Not a parametric alias, restored the consumed token and parse a target
    // list as usual.
    cursor.restore(p.getLexer());
  }

  ExprNode *targetList;
  LogicalResult ret = p.parseTargetListExpr(targetList, stmtIndent);
  result = targetList;
  return ret;
}

ParseResult StmtParser::parseAliasDeclStmtBody(LexerCursor startCursor,
                                               size_t stmtIndent, SMLoc kwLoc) {
  // The 'alias' or 'comptime' keyword was already consumed by the caller.
  Location loc = translateLocation(kwLoc);

  SmartVariant<StringRef, ExprNode *> parseResult;
  if (failed(parseAliasDeclTargetsExpr(*this, parseResult, stmtIndent)))
    return failure();

  if (auto targetExpr = dyn_cast<ExprNode *>(parseResult)) {
    if (!consumeIf(Token::equal)) {
      emitError(kwLoc, "expected '=' after comptime declaration");
      return failure();
    }

    // Reject the stmt if it is in a struct/trait, resolving them eagerly can
    // easily lead to parser cycles.
    if (isa_and_nonnull<StructDeclOp, TraitDeclOp>(
            parentDecl.getIfOperation())) {
      emitError(kwLoc,
                isa<StructDeclOp>(parentDecl.getIfOperation())
                    ? "'comptime' constants inside structs must be declared "
                      "separately; break this into individual declarations"
                    : "a trait's associated types must be declared separately; "
                      "break this into individual declarations");
      return failure();
    }

    // Parser the initializer pvalue and destructuring it eagerly.
    ExprNode *initExpr;
    if (parseVarInitExpression(initExpr, stmtIndent))
      return failure();
    IREmitter emitter = getParamEmitter(EC_AliasValue);
    PValue initPVal = emitter.emitExprPValue(initExpr, EC_AliasValue);
    if (!initPVal)
      return failure();

    return emitter.emitDestructuringPValue(initPVal, targetExpr);
  };

  // Else, do lazy resolution for a single target alias declaration.
  StringAttr name = builder.getStringAttr(cast<StringRef>(parseResult));

  // Before parsing the rest of the alias, the type is unresolved and value is
  // UnresolvedAliasValueAttr.
  auto type = UnresolvedType::get(getContext());

  // TODO(fixme): currently, we cannot rely on looking up name collisions of
  // aliases because of things like this:
  // def foo():
  //     def bar():
  //         alias z = __mlir_attr.`0: index`
  //     alias z = __mlir_attr.`1: index`
  // So we treat them as implicitly declared to force a mangling. We could
  // probably fix this when parameters stop being non-lexical.
  StringAttr mangledName = parentDecl.mangleParamName(name.strref());
  auto decl = ParamDeclAttr::get(mangledName, type);
  auto declOp = AliasDeclOp::create(builder, loc, decl);

  // Skip the body of this definition: go to a token the starts a line at the
  // same indent level (or less) as the current definition.
  bool isDefaultedAlias = false;
  llvm::unique_function<bool()> defaultAliasDetector = {};
  if (isa_and_nonnull<TraitDeclOp>(curDeclScope->getIfOperation())) {
    defaultAliasDetector = [&]() -> bool {
      // if we found one `=`
      if (getToken().is(Token::equal))
        isDefaultedAlias = true;
      // Keep parsing.
      return false;
    };
  }
  skipUntilIndentation(stmtIndent, /*stopOnSemicolon=*/true,
                       std::move(defaultAliasDetector));
  declOp.setDefaultedAssociatedAlias(isDefaultedAlias);
  // Skip the trailing docstring if it has one. We treat these as part of the
  // alias decl.
  (void)consumeIf(Token::string);

  // Remember that we parsed this declaration so we can finish type checking it
  // when it gets referenced.
  getDeclResolver().addDecl(declOp, kwLoc, name, curDeclScope, startCursor,
                            getLexer().getCursor(), stmtIndent);
  return success();
}

ParseResult StmtParser::parseStructStmt(LexerCursor startCursor,
                                        size_t curIndent) {
  // We don't support non-top level structs (yet?).
  bool nestFailure = false;
  if (isa_and_nonnull<StructDeclOp>(getParentDecl().getIfOperation())) {
    emitTokenError("nested struct not supported here");
    nestFailure = true;
  } else if (isa_and_nonnull<TraitDeclOp>(getParentDecl().getIfOperation())) {
    emitTokenError("nested struct in a trait not supported here");
    nestFailure = true;
  } else if (isa_and_nonnull<FnOp>(getParentDecl().getIfOperation())) {
    emitTokenError("struct inside a function not supported here");
    nestFailure = true;
  }
  consumeToken(Token::kw_struct);

  SMLoc smLoc;
  StringAttr nameAttr;
  if (parseIdentifier(nameAttr, "expected struct name", &smLoc,
                      /*forbidStartOfLine=*/true)) {
    return failure();
  }

  // The parameter list must begin on the same line as the struct name;
  // otherwise the decl-extent scan below truncates the declaration at the
  // start-of-line bracket, surfacing as a confusing internal error during
  // signature resolution.
  if (getToken().is(Token::l_square) &&
      rejectTokenAtStartOfLine("parameter list"))
    return failure();

  auto loc = translateLocation(smLoc);

  auto newStruct = StructDeclOp::create(builder, loc, nameAttr);

  // Skip the body of this definition: go to a token the starts a line at the
  // same indent level (or less) as the current definition.
  skipUntilIndentation(curIndent);

  if (nestFailure) {
    getDeclResolver().addErroneousDecl(nameAttr.getValue(), smLoc,
                                       curDeclScope);
  } else {
    // Remember that we parsed this declaration so we can finish type checking
    // it when it gets referenced.
    getDeclResolver().addDecl(newStruct, smLoc, nameAttr, curDeclScope,
                              startCursor, getLexer().getCursor(), curIndent);
  }
  return success();
}

ParseResult StmtParser::parseTraitStmt(LexerCursor startCursor,
                                       size_t curIndent) {
  // We don't support non-top level traits (yet?).
  if (!isa_and_nonnull<FileModuleOp>(getParentDecl().getIfOperation()))
    emitTokenError("nested trait not supported here");

  consumeToken(Token::kw_trait);

  SMLoc smLoc;
  StringAttr nameAttr;
  if (parseIdentifier(nameAttr, "expected trait name", &smLoc,
                      /*forbidStartOfLine=*/true))
    return failure();
  auto loc = translateLocation(smLoc);

  auto newTrait = TraitDeclOp::create(builder, loc, nameAttr);

  // Skip the body of this definition: go to a token the starts a line at the
  // same indent level (or less) as the current definition.
  skipUntilIndentation(curIndent);

  // Remember that we parsed this declaration so we can finish type checking it
  // when it gets referenced.
  getDeclResolver().addDecl(newTrait, smLoc, nameAttr, curDeclScope,
                            startCursor, getLexer().getCursor(), curIndent);
  return success();
}

ParseResult StmtParser::parseExtensionStmt(LexerCursor startCursor,
                                           size_t curIndent) {
  // We don't support non-top level extensions. One day this might be useful
  // to enable the user to mimic implicit trait conformance. But this is not
  // that day.
  bool nestFailure = false;
  if (isa_and_nonnull<StructDeclOp>(getParentDecl().getIfOperation())) {
    emitTokenError("nested extension not supported here");
    nestFailure = true;
  } else if (isa_and_nonnull<TraitDeclOp>(getParentDecl().getIfOperation())) {
    emitTokenError("nested extension in a trait not supported here");
    nestFailure = true;
  } else if (isa_and_nonnull<FnOp>(getParentDecl().getIfOperation())) {
    emitTokenError("extension inside a function not supported here");
    nestFailure = true;
  }
  consumeToken(Token::kw___extension);

  SMLoc nameSMLoc;
  // Unprefixed/unsuffixed name. `extension Spaceship`'s base name is Spaceship.
  StringAttr targetStructNameAttr;
  if (parseIdentifier(targetStructNameAttr, "expected extension name",
                      &nameSMLoc, /*forbidStartOfLine=*/true))
    return failure();

  auto loc = translateLocation(nameSMLoc);

  std::string nameStr = "extension:" + targetStructNameAttr.getValue().str();
  auto nameAttr = StringAttr::get(getContext(), nameStr);

  // Note that this is using the unique name, not the base name.
  // Further below, we'll still add it to the parent ASTDecl with the base name.
  // This is different than other constructs, where the ASTDecl knows them by
  // their true name.
  // TODO(MOCO-522): This is arcana, reference some central docs here and
  // everywhere else this comes into play
  auto extensionDeclOp =
      ExtensionDeclOp::create(builder, loc, nameAttr, targetStructNameAttr);

  // Skip the body of this definition: go to a token the starts a line at the
  // same indent level (or less) as the current definition.
  skipUntilIndentation(curIndent);

  if (nestFailure) {
    getDeclResolver().addErroneousDecl(targetStructNameAttr.getValue(),
                                       nameSMLoc, curDeclScope);
  } else {
    // The parent ASTDecl knows this ExtensionDeclOp by the name e.g.
    // "extension:MyStruct". This might not be unique; there can be multiple
    // extensions for the same struct.
    // TODO(MOCO-522): This is arcana, reference some central docs here and
    // everywhere else this comes into play
    ASTDecl &decl = getDeclResolver().addDecl(
        extensionDeclOp, nameSMLoc, nameAttr, curDeclScope, startCursor,
        getLexer().getCursor(), curIndent);

    // The extension should also be known to its parent as "extension:". Some
    // places look up that string when they want to find all extensions in a
    // particular scope.
    // TODO(MOCO-522): Arcana docs on this.
    StringAttr extensionsNameAttr = StringAttr::get(getContext(), "extension:");
    getDeclResolver().aliasDeclInParent(&decl, extensionsNameAttr);
  }
  return success();
}

ParseResult StmtParser::parseClassStmt(LexerCursor startCursor,
                                       size_t curIndent) {
  emitTokenError("classes are not supported yet");
  consumeToken(Token::kw_class).getLoc();

  // Skip the body of this definition: go to a token the starts a line at the
  // same indent level (or less) as the current definition.
  skipUntilIndentation(curIndent);
  return success();
}

/// An MLIR region declaration defines a single block region body as a suite
/// with the declaration arguments corresponding to the region arguments. It is
/// used to define regions for MLIR operations.
///
/// region_stmt ::= "__mlir_region" identifier "(" [argument_list] ")" ":" suite
ParseResult StmtParser::parseMLIRRegionStmt(LexerCursor startCursor,
                                            size_t curIndent) {
  SMLoc loc = consumeToken(Token::kw___mlir_region).getLoc();

  // We will be moving the builder into the contained region, so save it here.
  llvm::SaveAndRestore builderSaver(builder);

  // Resolve the signature and the body immediately.
  StringAttr identifier;
  if (parseIdentifier(identifier, "expected a region name") ||
      parseToken(Token::l_paren, "expected '(' for parameter list"))
    return failure();

  // Create the decl corresponding to the region declaration.
  auto op = UnboundRegionOp::create(builder, translateLocation(loc));
  ASTDecl &decl =
      getDeclResolver().addDecl(op, loc, identifier, curDeclScope, startCursor,
                                getLexer().getCursor(), curIndent);
  decl.resolvedness = DeclResolvedness::body;

  // Parse the argument list if present.
  struct RegionArgument {
    StringAttr name;
    SMLoc loc;
  };
  SmallVector<RegionArgument> args;
  SmallVector<Type> argTypes;
  SmallVector<Location> argLocs;
  if (!consumeIf(Token::r_paren)) {
    // Parse simple argument: MLIR operations don't have input conventions.
    auto parseArg = [&]() -> ParseResult {
      RegionArgument &arg = args.emplace_back();
      ExprNode *typeExpr;
      if (getLocation(arg.loc) ||
          parseIdentifier(arg.name, "expected an identifier") ||
          parseToken(Token::colon, "expected ':' after region argument") ||
          parseExpression(typeExpr))
        return failure();
      ASTType type = getEmitter().emitExprType(typeExpr);
      if (!type)
        return failure();
      argTypes.push_back(type);
      argLocs.push_back(translateLocation(arg.loc));
      return success();
    };
    if (parseCommaSeparatedList(parseArg, Token::r_paren))
      return failure();
    consumeToken(Token::r_paren);
  }

  builder.createBlock(&op.getRegion());
  for (auto [regionArg, parsedArg] :
       llvm::zip(op.getRegion().addArguments(argTypes, argLocs), args)) {
    // Add the declaration for the argument within the region declaration.
    getDeclResolver().addFullyResolvedDecl(SRValue(regionArg), parsedArg.name,
                                           parsedArg.loc, &decl);
  }

  if (parseToken(Token::colon, "expected ':' after region argument list"))
    return failure();
  StmtParser parser(lexer, decl);
  return parser.parseLocalScopeSuite(curIndent);
}

//===----------------------------------------------------------------------===//
// List/Dict/Set Comprehensions
//===----------------------------------------------------------------------===//

static LogicalResult emitIfClause(StmtParser &stmtEmitter,
                                  const ComprehensionClause &clause,
                                  std::function<LogicalResult()> callback) {
  auto location = stmtEmitter.translateLocation(clause.kwLoc);
  auto emitter = stmtEmitter.getEmitter();

  RValue condI1RVal = emitter.emitExprScalarBool(clause.expr, EC_BoolCondition);
  if (!condI1RVal)
    return failure();
  SRValue condRVal =
      emitter.emitSRValue({condI1RVal, clause.expr}, EC_BoolCondition);
  if (!condRVal)
    return failure();

  return success((bool)emitter.emitIfThen(
      location, condRVal, TypeRange(),
      [&]() -> LogicalResult {
        llvm::SaveAndRestore builderSaver(stmtEmitter.getBuilder());
        stmtEmitter.getBuilder().setInsertionPointToStart(
            emitter.builder->getInsertionBlock());
        if (failed(callback()))
          return failure();
        HLCF::YieldOp::create(*emitter.builder, location);
        return success();
      },
      [&]() -> LogicalResult {
        HLCF::YieldOp::create(*emitter.builder, location);
        return success();
      }));
}

/// Emit the clauses for a comprehension expression, then call the callback.
static LogicalResult
emitComprehensionsAnd(StmtParser &stmtEmitter,
                      ArrayRef<ComprehensionClause> clauses,
                      std::function<LogicalResult()> callback) {
  // If we ran out of clauses, then we're bottomed out at the callback.  Invoke
  // it and be done.
  if (clauses.empty())
    return callback();

  auto clause = clauses.front();
  switch (clause.kind) {
  case ComprehensionClause::kFor: {
    // Emit the for statement with a body the processes the rest of the clauses.
    auto forStmt = stmtEmitter.emitForStmt(
        clause.kwLoc, clause.forPattern, clause.expr, [&]() -> LogicalResult {
          return emitComprehensionsAnd(stmtEmitter, clauses.drop_front(),
                                       callback);
        });
    return success((bool)forStmt);
  }
  case ComprehensionClause::kIf:
    return emitIfClause(stmtEmitter, clause, [&]() -> LogicalResult {
      return emitComprehensionsAnd(stmtEmitter, clauses.drop_front(), callback);
    });
  }
  llvm_unreachable("unhandled comprehension clause kind");
}

/// Emit a comprehension expression into the specified emitter.  If a
/// contextual type is known, 'expectedType' is non-null.
static AnyValue emitComprehension(const ComprehensionNode *node, ExprDest &dest,
                                  IREmitter &emitter) {
  auto &shared = emitter.shared;
  auto loc = node->getLoc();
  auto location = shared.translateLocation(loc);

  // Comprehensions are more like a statement then they are an expression.
  // we can't emit them into PValue contexts until we have generalized PValue
  // support ala:
  // https://www.notion.so/modularai/Generalized-PValue-Support-62c85f77f13c4d9bad30e398f04ce1a9
  if (!emitter.builder) {
    emitter.emitError(loc,
                      "list comprehension must execute in runtime contexts; "
                      "remove 'comptime' and move this into a function body")
        << node->getRange();
    return {};
  }

  // The general structure we emit for '[x*x for x in range(10) if x != 4]' is:
  //   var result = Collection()
  //   for x in range(10):
  //     if x != 4:
  //       result.append(x*x)
  //   "return" result
  //
  // The `Collection` is a type that implements `__init__` and `append`. It can
  // be inferred from context, but otherwise defaults to List[T] (where T is the
  // element type of x*x).  This is a bit awkward because we cannot know the
  // type of 'result' until we emit all the other stuff.
  const char *emptyString = ""; // make sure we have a nul on this.
  Lexer lexer(shared.diags, emptyString, emptyString);
  StmtParser stmtEmitter(lexer, emitter);

  // We're going to emit a bunch of stuff below but need a cursor to know where
  // to put the temporary for the collection and the constructor call.  Emit a
  // temporary operation so we can find it later.
  auto cursor = LIT::ReturnOp::create(stmtEmitter.getBuilder(), location,
                                      ArrayRef<Value>());
  IREmitter cursorEmitter(emitter.declScope, OpBuilder(cursor));
  DebugInfo::DIBuilder cursorDIBuilder(emitter.getContext());
  if (shared.diBuilder)
    cursorDIBuilder = shared.diBuilder->copy();

  // Start out the result collection with an inferred type if it isn't known.
  auto inferCollectionType = [&](ASTType eltType, ASTType valType) -> ASTType {
    // Use the contextual type if it is known.
    if (auto expectedType = dest.getExpectedTypeIfSpecified())
      return expectedType;

    // Otherwise default to List[T], Set[T] or Dict[K, V].
    ASTType defaultType;
    if (node->kind == ExprNode::kListComprehension)
      defaultType = shared.getStandardCollectionType(loc, "List");
    else if (node->kind == ExprNode::kSetComprehension)
      defaultType = shared.getStandardCollectionType(loc, "Set");
    else if (node->kind == ExprNode::kDictComprehension)
      defaultType = shared.getStandardCollectionType(loc, "Dict");
    else
      llvm_unreachable("unhandled comprehension kind");
    if (!defaultType)
      return {};

    // Form T[eltType] syntactically and emit it.
    SyntheticNode collTypeExpr(loc, PValue(defaultType));
    SyntheticNode eltTypeExpr(loc, PValue(eltType));
    SyntheticNode valTypeExpr(loc, PValue(valType));
    Operand subscriptOperand[] = {
        {&eltTypeExpr, loc, ArgUnpackStyle::kPositional},
        {&valTypeExpr, loc, ArgUnpackStyle::kPositional}};
    SubscriptNode subscript(&collTypeExpr, loc,
                            node->kind == ExprNode::kDictComprehension
                                ? ArrayRef<Operand>(subscriptOperand)
                                : ArrayRef<Operand>(subscriptOperand[0]),
                            loc);
    return stmtEmitter.getParamEmitter(EC_CollectionCompElt)
        .emitExprType(&subscript);
  };

  // Dummy node with the correct location.
  SyntheticNode exprNode(loc);
  MLValue collectionMLValue;

  // This emits the body of the comprehension in the context of the clauses.
  auto emitBody = [&]() -> LogicalResult {
    auto emitter = stmtEmitter.getEmitter();
    auto elementExpr = emitter.emitExprCValue(node->expr, EC_CollectionCompElt);
    if (!elementExpr)
      return failure();

    CValue valueExpr;
    ASTType valueType;
    if (node->kind == ExprNode::kDictComprehension) {
      valueExpr = emitter.emitExprCValue(node->valueExpr, EC_CollectionCompElt);
      if (!valueExpr)
        return failure();
      valueType = valueExpr.getRValueType();
    }

    // If we had no expected type, then assign the default collection type now.
    auto collectionType =
        inferCollectionType(elementExpr.getRValueType(), valueType);
    if (!collectionType)
      return failure();

    // Now that we know the collection type, we can materialize the temporary
    // with the right type (which might also reuse an existing buffer). Note
    // that this uses cursorEmitter so the temp gets emitted to the right spot.
    {
      // Make sure any synthesized declarations have the right debug info scope
      // from the cursor position.
      std::unique_ptr<DebugInfo::DIBuilder> newDIBuilder;
      if (shared.diBuilder)
        newDIBuilder = std::make_unique<DebugInfo::DIBuilder>(cursorDIBuilder);

      llvm::SaveAndRestore keep(shared.diBuilder, std::move(newDIBuilder));
      collectionMLValue =
          dest.getMLValueForResult(loc, collectionType, cursorEmitter);
    }

    // Okay we know the collection has a type, emit the insertion method call.
    // Use dict.__setitem__(key, value) for dict comprehensions.
    if (node->kind == ExprNode::kDictComprehension) {
      CallOperands operands(CallSyntax::kMethodCall, &exprNode,
                            EC_CollectionCompElt,
                            {{collectionMLValue, &exprNode},
                             {elementExpr, &exprNode},
                             {valueExpr, &exprNode}});
      operands.hasSelfOperand = true;
      emitter.emitNamedMethodCall("__setitem__", std::move(operands));
    } else {
      // Use list.append(elt) or set.add(elt) for list and set comprehensions.
      CallOperands operands(
          CallSyntax::kMethodCall, &exprNode, EC_CollectionCompElt,
          {{collectionMLValue, &exprNode}, {elementExpr, &exprNode}});
      operands.hasSelfOperand = true;
      const char *name =
          node->kind == ExprNode::kListComprehension ? "append" : "add";
      emitter.emitNamedMethodCall(name, std::move(operands));
    }

    return success();
  };

  if (failed(emitComprehensionsAnd(stmtEmitter, node->clauses, emitBody)))
    return {};

  // Leave the caller's IREmitter at the right place to continue emitting.
  emitter.builder = stmtEmitter.getBuilder();

  // Now that we know we have the collection type, emit the call to the
  // initializer at the cursor.
  ExprDest ctorDest(collectionMLValue, EC_CollectionCompElt);
  cursorEmitter.emitConstructorCall(
      collectionMLValue.getRValueType(),
      CallOperands(CallSyntax::kTypeCall, &exprNode, std::move(ctorDest)));

  // Finally, we can nuke the cursor.
  cursor->erase();

  // Success! Return ownership of the result expression.
  return MRValue(collectionMLValue);
}

AnyValue ComprehensionNode::emitIR(ExprDest &dest, IREmitter &emitter) const {
  // emitComprehension is defined in the statement emission file.
  AnyValue result = emitComprehension(this, dest, emitter);
  return emitter.emitResult(result, this, dest);
}

//===----------------------------------------------------------------------===//
// Entry point to this file
//===----------------------------------------------------------------------===//

/// Parse a 'suite' production into the declaration specified by `ASTDecl`.
/// This is the main entrypoint to this file.
ParseResult ParserBase::parseSuite(ASTDecl &containingDecl) {
  StmtParser parser(lexer, containingDecl);

  // Parse the docstring if present.
  parser.parseDocString(containingDecl);

  // Parse the remaining body of the declaration.
  return parser.parseSuite(containingDecl.getIndentation());
}
