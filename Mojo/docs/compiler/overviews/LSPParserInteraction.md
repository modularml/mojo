# Mojo LSP ↔ Parser Interaction

A working reference for how the Mojo language server (LSP) interacts with the
Mojo parser. This is a living document; sections are added as we explore.

Scope: **text documents (`.mojo` files) only.** Notebook/REPL handling is
intentionally out of scope and omitted throughout.

## Table of contents

- [Overview](#overview)
- [The seam: `MojoParserContext`](#the-seam-mojoparsercontext)
- [The unifying model: parse + listener + sink](#the-unifying-model-parse--listener--sink)
- [Generalized parse: indexing the document](#generalized-parse-indexing-the-document)
  - [Lifecycle and context](#lifecycle-and-context)
  - [Underlying parse: `parseFileForLSP` vs `importMojoFile`](#underlying-parse-parsefileforlsp-vs-importmojofile)
  - [Sink 1: the `SymbolIndex` (write side)](#sink-1-the-symbolindex-write-side)
  - [Sink 2: diagnostics](#sink-2-diagnostics)
  - [`checkModuleSemantics`: feeding back semantic diagnostics](#checkmodulesemantics-feeding-back-semantic-diagnostics)
- [Specialized parses: answering cursor queries](#specialized-parses-answering-cursor-queries)
  - [Common shape](#common-shape)
  - [`codeComplete`](#codecomplete)
  - [`signatureHelp`](#signaturehelp)
  - [Docstring code blocks: `parseREPLExpression` (the exception)](#docstring-code-blocks-parsereplexpression-the-exception)
- [Reference: the full `ParserListener` interface](#reference-the-full-parserlistener-interface)
- [Threading: using a single-threaded parser safely](#threading-using-a-single-threaded-parser-safely)
- [Follow ups](#follow-ups)

## Overview

The LSP server is a thin language-feature layer wrapped around the *real* Mojo
parser/type-checker. It never reimplements language logic — every feature
(hover, completion, diagnostics, etc.) is derived from data the actual parser
produces.

Three-layer structure:

```text
LSPServer.cpp        JSON-RPC method names  <->  C++ handlers
   |
MojoServer           document map, debouncer, dispatch by URI
   |
MojoDocument         <- where the parser is invoked
   |__ Context { MLIRContext, ParserConfig, MojoParserContext,
                 SymbolIndex, LSPParserListener }
```

Each open file is a `MojoTextDocument` (a subclass of `MojoDocument`). Every
document owns a `Context` that bundles a fresh `MLIRContext`, a `ParserConfig`,
the `MojoParserContext`, and the two bridge objects (`SymbolIndex` +
`LSPParserListener`).

Key source files:

| File                                           | Role                                                                                                        |
|------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| `KGEN/tools/mojo-lsp-server/MojoServer.cpp`    | Bulk of the implementation: `Context`, `LSPParserListener`, `SymbolIndex`, parse flow, all feature handlers |
| `KGEN/tools/mojo-lsp-server/MojoDocument.h`    | Document model headers, feature API                                                                         |
| `Mojo/include/Mojo/MojoTooling/ParserDriver.h` | `MojoParserContext` — the tooling-facing parser driver                                                      |
| `Mojo/include/Mojo/MojoParser/EntryPoint.h`    | `ParserConfig` and the `ParserListener` interface                                                           |

## The seam: `MojoParserContext`

The boundary between the LSP and the parser is one class, `MojoParserContext`
(`Mojo/lib/MojoTooling/`), the tooling-facing driver that owns parser state. The
LSP holds one per open document and drives it.

It is constructed inside `MojoDocument::Context`, which also wires the LSP's
listener into the `ParserConfig`:

```728:749:KGEN/tools/mojo-lsp-server/MojoServer.cpp
struct MojoDocument::Context {
  Context(MojoDocument &mainDoc)
      : mlirContext(MLIRContext::Threading::DISABLED),
        parserConfig(&mlirContext, compilationOptions), symbolIndex(mainDoc),
        parserListener(mainDoc, symbolIndex) {
    parserConfig.parserListener = &parserListener;
    ...
    parserContext =
        std::make_unique<MojoParserContext>(mainDoc.sourceMgr, parserConfig);
  }
```

## The unifying model: parse + listener + sink

The LSP drives the parser the **same way** for every feature: it invokes a parse
and installs a `ParserListener` that the parser calls back during resolution,
gated by `isInterestedInLoc`. There is only one parser and (for text documents)
one parse routine — `parseFileForLSP`. A feature is fully determined by **four
knobs**:

1. **which listener** (i.e. which `ParserListener` hooks it overrides),
2. **the location filter** (`isInterestedInLoc`),
3. **the sink** (where results go), and
4. **the context lifetime** (persistent vs throwaway).

Concretely, each LSP request maps to a parser entry point on `MojoParserContext`
(all of which — except docstring code blocks — run `parseFileForLSP`
underneath):

| Request              | Parser entry point                                        | Persona                            |
|----------------------|-----------------------------------------------------------|------------------------------------|
| Document open/edit   | `parseFileForLSP`                                         | generalized                        |
| Completion           | `codeComplete(buffer, pos, …)` (fresh re-parse at cursor) | specialized                        |
| Signature help       | `signatureHelp(buffer, pos, …)`                           | specialized                        |
| Docstring code block | `parseREPLExpression`                                     | specialized (exception — see TODO) |

Those knobs collapse into two personas:

| Knob             | Generalized (indexing) parse                                    | Specialized (query) parses                      |
|------------------|-----------------------------------------------------------------|-------------------------------------------------|
| Trigger          | `didOpen`/`didChange` → debounce → async                        | a feature request at a cursor                   |
| Context lifetime | **persistent** (document's `Context`)                           | **throwaway** (fresh `SourceMgr`/`MLIRContext`) |
| Listener         | `LSPParserListener` (broad: decls + refs)                       | feature-specific (narrow hook pair)             |
| Location filter  | whole main file                                                 | the cursor range/buffer                         |
| Sink             | persistent `SymbolIndex` **+ diagnostics → publishDiagnostics** | transient result buffer, then discarded         |
| Diagnostics      | collected & published                                           | **suppressed**                                  |
| Frequency        | once per edit (debounced)                                       | full re-parse per request                       |

The two personas consume **disjoint** sets of listener hooks (only
`isInterestedInLoc` is shared, with a different meaning each time):

| Hook                                                                                                                                                       | Generalized (doc parse) |    Completion    |  Signature help   |
|------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------:|:----------------:|:-----------------:|
| `isInterestedInLoc`                                                                                                                                        |        main file        | completion range | completion buffer |
| `onAliasDecl`, `onFunctionDecl`, `onStructDecl`, `onStructFieldDecl`, `onTraitDecl`, `onVariableDecl`, `onParameterDecl`, `onArgumentDecl`, `onModuleDecl` |       ✓ (→ index)       |        —         |         —         |
| `onRef`, `onModuleImport`                                                                                                                                  |       ✓ (→ index)       |        —         |         —         |
| `onMemberLookup`, `onImport`                                                                                                                               |            —            |        ✓         |         —         |
| `onCall`, `onParameterBinding`                                                                                                                             |            —            |        —         |         ✓         |

The key fact: **the four hooks the generalized listener leaves as no-ops
(`onMemberLookup`, `onImport`, `onCall`, `onParameterBinding`) are exactly the
ones the specialized listeners consume.** This reflects a split in what the
hooks *mean*:

- **Generalized hooks = structural facts**: "this declaration exists here,"
  "this reference binds to that decl." Stable, whole-file, location-anchored. →
  builds the persistent `SymbolIndex`.
- **Specialized hooks = in-flight resolution events at a point**: "a member is
  being looked up on X," "an import is resolving here," "a call is being
  resolved with these operands." They fire *during* the act of resolving an
  expression and expose the candidate set / scope at a position — exactly what a
  cursor-driven feature needs. → builds a transient result.

The remainder of the doc details the generalized parse (its parse routine and
its two sinks) and then the specialized parses (their shared shape and each
variant).

## Generalized parse: indexing the document

The generalized parse runs when a file is opened or edited. Its job is to build
the persistent per-document state — the `SymbolIndex` — and to publish
diagnostics. It uses the document's persistent `Context` (and thus its
`LSPParserListener`, wired in [the seam](#the-seam-mojoparsercontext)).

### Lifecycle and context

Trigger and flow:

```text
didOpen / didChange
  → debounce (150 ms)
  → startDocumentParse → async task → parseDocument
       → context = make_unique<Context>(*this)   // fresh SymbolIndex + LSPParserListener
       → parseDocumentImpl → parseFileForLSP(mainFileID)
       → diagnostics published
```

The parse is async (on the document's `AsyncValueRef<Chain>`); subsequent
feature requests are enqueued *behind* it in FIFO order, so they transparently
see the freshly built index. The document parse entry point itself is one call
into the tooling driver, followed by semantic checking and docstring handling:

```1940:1951:KGEN/tools/mojo-lsp-server/MojoServer.cpp
size_t MojoTextDocument::parseDocumentImpl() {
  KGEN::CompilerTimeTraceScope traceScope("parseTextDocument");

  parsedDecl =
      getParserContext().parseFileForLSP(getSourceMgr().getMainFileID());

  checkModuleSemantics(parsedDecl);
  processDocStrings(docStrings, parsedDecl);
  getParserContext().ensureSignaturesResolved();

  return contents.length();
}
```

(`checkModuleSemantics` runs the LIT "check" pipeline on a clone and is a
separate thread — see [Follow ups](#follow-ups). `processDocStrings` handles
docstring code blocks, also TODO.)

> Because the `SymbolIndex` lives in the per-parse `Context`
> (`context = std::make_unique<Context>(*this)`), every document parse
> **discards and rebuilds the index from scratch** — there is no incremental
> update.

### Underlying parse: `parseFileForLSP` vs `importMojoFile`

Both share the same starting point (`buildModuleDecl` builds the top-level
module decl). They diverge entirely in **how deeply they resolve** and **what
post-processing they run**.

#### Background: leveled, lazy resolution

The parser resolves on demand. Each `ASTDecl` tracks its resolution level
(`Mojo/include/Mojo/MojoParser/SharedState.h`):

```text
DeclResolvedness:  unparsed  ->  signature  ->  body
```

- `unparsed` — only the identifier is known.
- `signature` — params/metaparams parsed and type-checked; body not processed.
- `body` — fully type-checked, including the body.

`resolveBody(decl, loc)` and `resolveSignature(decl, loc)` are thin wrappers
over `resolve(decl, body|signature, loc)`.

#### The regular path: `importMojoFile` -> `importMojoImpl`

Goal: produce a complete, canonical, verified IR module for the compiler. Steps
(`Mojo/lib/MojoParser/EntryPoint.cpp`, `importMojoImpl`):

1. `resolveAllReferencedFrom(moduleDecl)` — two stages, both **full body**:
   - Stage 1: BFS `resolveBody` on all decls within the main module.
   - Stage 2: `resolveReferencedDecls()` iteratively `resolveBody`s the **entire
     transitive closure** of referenced external/stdlib decls.
2. `finalizeImportedBytecodeModules()` — drop never-referenced bytecode ops.
3. **Bail to null** if any error was emitted.
4. `mlir::verify(*module)` — full structural verification (non-production).
5. `eraseUnreachableDecls(...)` — symbol DCE; strip imports/aliases/dead symbols
   to canonical IR (also stabilizes the compile-cache key).
6. `sortValueUses(*module)` — deterministic use-list order for stable bytecode.
7. Returns `OwningOpRef<ModuleOp>` (owned IR).

#### The LSP path: `parseFileForLSP`

Goal: just enough resolved AST to answer editor queries, cheaply, tolerating
broken code. Entire body (`Mojo/lib/MojoTooling/ParserDriver.cpp`):

```257:269:Mojo/lib/MojoTooling/ParserDriver.cpp
MojoASTDeclRef MojoParserContext::parseFileForLSP(unsigned fileId) {
  ...
  ASTDecl *moduleDecl = buildModuleDecl(filepath, sourceBuf, impl->sharedState);
  resolveForLSP(*impl->sharedState.declResolver, *moduleDecl);
  resolveSignaturesForLSP(*impl->sharedState.declResolver);

  return MojoASTDeclRef(moduleDecl);
}
```

- `resolveForLSP` — BFS `resolveBody` on decls **descended from the main file's
  root only** (plus `validateDocString`). Crucially it does **not** call
  `resolveReferencedDecls()`, so external/stdlib bodies are never pulled in.
- `resolveSignaturesForLSP` — walks everything touched (`parsedDeclList`,
  including referenced externals) and resolves each to **signature depth only**,
  with three carve-outs:
  - lazy named imports (`UnresolvedImportOp`, still `unparsed`) are **skipped**
    to avoid pulling transitive imports;
  - imported bytecode `FnOp`s get `resolveSignature` only (bodies already
    materialized in IR);
  - other bytecode container decls get `resolveBody` (needed to satisfy the LLVM
    invariant that SymbolTable containers have a non-empty body region).

#### Side-by-side

| Aspect                            | `importMojoFile` (compiler)        | `parseFileForLSP` (LSP)            |
|-----------------------------------|------------------------------------|------------------------------------|
| Main-file decls                   | full **body**                      | full **body**                      |
| Referenced external/stdlib decls  | full **body** (transitive closure) | **signature only**                 |
| Imported bytecode `FnOp` bodies   | resolved                           | **left as empty stubs**            |
| Lazy named imports                | resolved                           | **skipped**                        |
| On parse error                    | returns **null**, discards IR      | returns the decl **anyway**        |
| `mlir::verify`                    | yes (non-prod)                     | **no**                             |
| `eraseUnreachableDecls` (DCE)     | yes                                | **no** (LSP needs imports/aliases) |
| `sortValueUses` (stable bytecode) | yes                                | **no**                             |
| `finalizeImportedBytecodeModules` | inline, checked                    | deferred to context destructor     |
| Return type                       | `OwningOpRef<ModuleOp>` (owned IR) | `MojoASTDeclRef` (decl handle)     |
| `ParserListener`                  | typically null                     | `LSPParserListener` (symbol index) |

#### Why the differences exist

1. **Speed:** skipping body resolution of the transitive stdlib closure is the
   dominant win — editing a file that imports `collections` must not type-check
   all of `collections`.
2. **Error tolerance:** editor code is usually mid-edit; the LSP returns a
   partial AST even after errors, while the compiler refuses to proceed.
3. **The LSP needs what the compiler discards:** imports, aliases, and
   unreferenced symbols (removed by `eraseUnreachableDecls`) are exactly what
   powers go-to-definition on imports, hover on unused aliases, and completion
   of not-yet-referenced names — so the LSP skips DCE.
4. **No bytecode output:** `sortValueUses` and `mlir::verify` exist to produce
   stable/valid bytecode; the LSP never serializes, so it skips both.

#### Downstream consequence

Because imported `FnOp` bodies are left as empty stubs, when the LSP later
clones the module to run the "check" pipeline (`checkModuleSemantics`), it must
mark those stubs `external` — a case the compiler path never hits, since it
erases stubs before the pipeline (`cloneDeclModuleForCompilation` in
`Mojo/lib/MojoParser/EntryPoint.cpp`, ~590–600).

### Sink 1: the `SymbolIndex` (write side)

This is the structural-facts sink of the generalized parse: the
`LSPParserListener` turns the parser's decl/ref callbacks into the persistent
`SymbolIndex`.

> Scope: this covers only the **write side** — how the parser pushes information
> into the server. How feature handlers later *read* the `SymbolIndex` to answer
> requests (hover, definition, references, rename, doc symbols, semantic tokens)
> is out of scope for this doc, since those reads do not involve the parser. The
> exceptions that *do* re-invoke the parser (`codeComplete`, `signatureHelp`)
> are the specialized parses below.

Files: `KGEN/tools/mojo-lsp-server/MojoServer.cpp` (`LSPParserListener`,
`SymbolIndex`, `Symbol`, `SymbolRef`), `Mojo/lib/MojoParser/SharedState.cpp`
(the `notifyListenerOnXxx` methods).

Key facts up front:

- Writes happen **synchronously, interleaved with parsing**. By the time
  `parseFileForLSP` returns, the index is fully built.
- The `SymbolIndex` lives in the per-parse `Context`, so each document parse
  **discards and rebuilds it from scratch** (no incremental update).

#### Where the parser calls the listener

The parser never calls the LSP listener directly. It calls
`SharedState::notifyListenerOnXxx(...)` at each resolution point, each gated by
an interest check:

```2280:2299:Mojo/lib/MojoParser/SharedState.cpp
static bool isListenerInterestedInLoc(ParserListener *listener, SMLoc loc) {
  return listener && listener->isInterestedInLoc(loc);
}
...
  if (isListenerInterestedInLoc(parserListener, identifierLoc))
    parserListener->onFunctionDecl(&decl, identifierLoc);
```

`LSPParserListener::isInterestedInLoc` accepts only main-file locations, so the
parser skips even materializing callback arguments for library code:

```568:571:KGEN/tools/mojo-lsp-server/MojoServer.cpp
  bool isInterestedInLoc(SMLoc parserLoc) override {
    return mainDoc.containsLocation(mainDoc.translateParserLoc(parserLoc));
  }
```

(`translateParserLoc` is identity for a plain text doc; it only remaps docstring
code-block locations into the main buffer.)

#### Two operations, many hooks

All declaration hooks funnel through `addSymbolDecl` -> `registerSymbol`; the
two reference hooks funnel through `registerRef`:

| Hooks                                                                                                                                                                                                         | Operation        |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------|
| `onAliasDecl`, `onFunctionDecl`, `onStructDecl`, `onStructFieldDecl`, `onTraitDecl`, `onVariableDecl`, `onParameterDecl`, `onArgumentDecl` (carries `argName`), `onModuleDecl` (skips main file's own module) | `registerSymbol` |
| `onRef` (resolved reference), `onModuleImport` (module name in `from M import …`)                                                                                                                             | `registerRef`    |

```604:611:KGEN/tools/mojo-lsp-server/MojoServer.cpp
void LSPParserListener::addSymbolDecl(ASTDecl *decl, SMLoc loc,
                                      std::optional<StringRef> identifier) {
  MojoASTDeclRef declRef(decl);
  if (!identifier)
    identifier = declRef.getName();
  symbolIndex.registerSymbol(declRef, identifier,
                             mainDoc.translateParserLoc(loc));
}
```

#### `Symbol` vs `SymbolRef`

The index separates a **declared entity** from an **occurrence**:

- `Symbol` — a declared entity, keyed by identity (`ASTDecl*`). Holds identifier
  text, `declRef`, approximate kind, declaration range, and a back-list of
  `SymbolRef`s. Modules/packages get a zero-length range; everything else gets
  `[loc, loc + identifier.size())`.
- `SymbolRef` — an occurrence: a source range pointing at **one or more**
  `Symbol`s (multiple because an overloaded name resolves to several decls).

A declaration's own identifier is *also* inserted as a `SymbolRef` pointing at
its own `Symbol` — that is what gives the declaration site an entry in the
interval map.

**A `Symbol` does not copy the decl's info — it borrows a pointer.**
`MojoASTDeclRef` wraps a raw `ASTDecl*` (a live MLIR `Operation*` in the
parser's module). The constructor captures only four cheap things: the
`identifier` string, the `declRef` pointer, an `approximateViewKind`, and an
arithmetic `range`. This is why creating a `Symbol` for a stdlib decl on
reference is cheap:

- **Kind** comes from `declRef.getApproximateDeclKind()` — a fast `isa<>` switch
  on the decl's operation that returns just the kind enum, never building a
  view.
- **Detailed info** (signature snippet, argument list, docstring, markdown) is
  the "correct but slow path": built lazily via `declRef.getDecl()` (a full
  `PublicDecl` view) only when a feature needs it (e.g. hover). Both share
  `getDeclImpl<ResultType>()` in `Mojo/lib/MojoTooling/MojoASTDeclRef.cpp`.
- **Required at creation:** only a non-empty name (`registerSymbol` bails on an
  empty identifier). The kind is computed but cheap; the detailed view is never
  computed at creation.

Two consequences:

- *Lifetime:* the `declRef` borrows into IR owned by the per-parse
  `MojoParserContext`; the `Symbol`/`SymbolIndex` live in the same `Context`, so
  the pointer is valid for the index's whole life and both are rebuilt together
  next parse.
- *Stdlib definition targets:* a referenced stdlib decl's `Symbol.range` points
  into the **stdlib buffer** (its real declaration location), since
  `ref.getLoc()` is outside the main file. `registerSymbol`'s `containsLocation`
  check keeps that range out of the main-doc interval map, but the location is
  retained on the `Symbol` — which is what lets go-to-definition jump into
  stdlib source.

Range helper (half-open span over the identifier text):

```71:75:KGEN/tools/mojo-lsp-server/MojoServer.cpp
static SMRange getRangeForText(SMLoc loc, StringRef text) {
  if (!loc.isValid())
    return {};
  return {loc, SMLoc::getFromPointer(loc.getPointer() + text.size())};
}
```

#### `registerSymbol`, `registerRef`, `insertRangeInMainDoc`

`registerSymbol` — dedup by decl identity into `symbolTable`, then add to the
interval map only if the range is in the main file and the decl is not a
module/package:

```360:383:KGEN/tools/mojo-lsp-server/MojoServer.cpp
Symbol *SymbolIndex::registerSymbol(MojoASTDeclRef declRef,
                                    std::optional<StringRef> identifier,
                                    SMLoc identifierLoc) {
  if (!identifier.has_value() || identifier->empty())
    return nullptr;
  auto [it, _] = symbolTable.try_emplace(
      &*declRef, std::make_unique<Symbol>(declRef, *identifier, identifierLoc));
  Symbol &symbol = *it->second;
  if (mainDoc.containsLocation(symbol.range.Start) &&
      mainDoc.containsLocation(symbol.range.End)) {
    if (!isa_and_nonnull<FileModuleOp, PackageOp>(declRef->getIfOperation()))
      insertRangeInMainDoc({symbol, symbol.range});
  }
  return &symbol;
}
```

`registerRef` — the bridge from main-file usage to a declaration. Reuse an
existing `Symbol` if the decl was already declared; otherwise register one on
the fly. This is how a main-file reference to a **stdlib/imported decl** (whose
declaration lives outside the main file) still gets a `Symbol`:

```393:419:KGEN/tools/mojo-lsp-server/MojoServer.cpp
  SmallVector<Symbol *> symbols;
  for (MojoASTDeclRef ref : declRefs) {
    if (Symbol *symbol = findSymbol(ref)) {
      symbols.push_back(symbol);
      continue;
    }
    std::optional<StringRef> symName = ref.getName();
    if (!symName) { /* DK_PublicArgumentDecl edge case: use spelling */ }
    if (Symbol *symbol = registerSymbol(
            ref, symName, mainDoc.translateParserLoc(ref.getLoc())))
      symbols.push_back(symbol);
  }
  if (!symbols.empty())
    insertRangeInMainDoc({symbols, range});
```

`insertRangeInMainDoc` — the `IntervalMap` write, with two rules: (1) refinement
— if a mapping already exists at the exact range with *more* symbols, replace it
with the smaller (more specific) set as the parser narrows an overload; (2)
first-writer-wins for overlaps — `IntervalMap` cannot store overlapping
intervals, so an overlapping new range is dropped. It also ignores the phantom
`__init__` module symbol at the start of `__init__.mojo` buffers.

#### Storage and interaction shape

```281:293:KGEN/tools/mojo-lsp-server/MojoServer.cpp
  using MapT = llvm::IntervalMap<
      SMLoc, SymbolRef *,
      llvm::IntervalMapImpl::NodeSizer<SMLoc, Symbol *>::LeafSize,
      llvm::IntervalMapHalfOpenInfo<SMLoc>>;
  ...
  MapT rangeToSymbolRef;                                  // range → occurrence
  SmallVector<std::unique_ptr<SymbolRef>> symbolRefs;     // owns SymbolRefs
  llvm::MapVector<ASTDecl *, std::unique_ptr<Symbol>> symbolTable; // decl → Symbol
```

Interaction characteristics:

- **Synchronous + interleaved** with parser resolution; no post-pass.
- **Two-stage main-file filter:** `isInterestedInLoc` (fast gate) plus the
  authoritative `containsLocation` checks in `registerSymbol`/`registerRef` (the
  interval map only ever holds main-file ranges).
- **Out-of-file decls enter only by reference:** a stdlib `Symbol` exists only
  because a main-file `SymbolRef` pulled it in.
- **Order-tolerant:** decls are usually pushed before references to them, but
  `registerRef`'s resolve-or-register fallback removes any strict ordering
  requirement.
- **Rebuilt per parse:** the index is owned by the per-parse `Context`.

### Sink 2: diagnostics

The second sink of the generalized parse is the diagnostic stream. Diagnostics
do **not** flow through the `ParserListener`; the parser emits them through
LLVM's `SourceMgr` diagnostic handler, which the LSP installs around the parse:

```808:822:KGEN/tools/mojo-lsp-server/MojoServer.cpp
        auto handlerFn = [](const llvm::SMDiagnostic &diag, void *ctx) {
          auto &handlerCtx = *static_cast<DiagHandlerContext *>(ctx);

          // If this is a note, add it to the last diagnostic group.
          if (diag.getKind() == llvm::SourceMgr::DK_Note) {
            if (!handlerCtx.smDiagnostics.empty())
              handlerCtx.smDiagnostics.back().push_back(diag);
            return;
          }
          // Remember errors found during parsing.
          if (diag.getKind() == llvm::SourceMgr::DK_Error)
            handlerCtx.doc.hasParserErrors = true;

          handlerCtx.smDiagnostics.push_back({diag});
        };
```

What flows here:

- `DK_Error` / `DK_Warning` / `DK_Remark` -> start a new diagnostic group.
- `DK_Note` -> appended to the *preceding* group (notes attach to their parent).
- Side effect: any `DK_Error` flips `hasParserErrors = true`, which later
  suppresses the heavier `checkModuleSemantics` LIT pipeline.

These become `textDocument/publishDiagnostics` notifications, and any fix-its
ride along in the same diagnostic and become **code actions**.

The two sinks of the generalized parse are orthogonal:

|                  | `ParserListener` (Sink 1)                                           | Diagnostic handler (Sink 2)                  |
|------------------|---------------------------------------------------------------------|----------------------------------------------|
| **Carries**      | AST structure: decls + references                                   | Problems: errors/warnings/notes + fix-its    |
| **Payload type** | `ASTDecl*` + `SMLoc`/`SMRange` + spelling                           | `llvm::SMDiagnostic`                         |
| **Direction**    | parser -> `SymbolIndex`                                             | parser -> `SourceMgr` handler -> list        |
| **Wired via**    | `parserConfig.parserListener`                                       | `sourceMgr.setDiagHandler(...)`              |
| **Powers**       | hover, definition, references, rename, doc symbols, semantic tokens | diagnostics, code actions (fix-its)          |
| **Lifetime**     | the `Context`/`SymbolIndex` (persists after parse)                  | scoped to the parse (`scope_exit` clears it) |

Mental model: the **listener** is a push API for AST facts the parser is
confident about ("this identifier IS a function named `foo` declared here"),
while the **diagnostic handler** is a push API for messages ("this identifier is
undefined"). A single parse fires many listener events and zero-or-more
diagnostics independently. (Specialized parses suppress diagnostics entirely, so
Sink 2 is unique to the generalized parse.)

### `checkModuleSemantics`: feeding back semantic diagnostics

After `parseFileForLSP`, `parseDocumentImpl` calls `checkModuleSemantics`, which
runs a few early compiler passes. **The only thing it feeds back to the LSP is
diagnostics** — and it does so by reusing Sink 2 (the same `SourceMgr` diag
handler), not a new channel. No structural info escapes it.

```926:952:KGEN/tools/mojo-lsp-server/MojoServer.cpp
void MojoDocument::checkModuleSemantics(MojoASTDeclRef decl) {
  ...
  // Don't check the semantics of the module if there were parser errors.
  if (hasParserErrors || !decl || !decl.getIfOperation())
    return;

  // Clone the module this decl is in so that we don't mess with the AST, as
  // this is used for other LSP queries.
  OwningOpRef<ModuleOp> tempModuleOp = cloneDeclModuleForCompilation(*decl);

  // Build a wrapper diagnostic handler for the source manager to capture
  // diagnostics emitted when processing the module.
  mlir::SourceMgrDiagnosticHandler sourceMgrDiagHandler(
      sourceMgr, tempModuleOp->getContext());

  KGEN::KGENCompiler kgenCompiler(*tempModuleOp->getContext(),
                                  getCompilationOptions());
  if (failed(kgenCompiler.runCheckLITPipeline(*tempModuleOp))) { ... }
}
```

#### The passes (`buildCheckLITPipeline`)

```20:33:Mojo/lib/Compiler/Pipeline/Pipeline.cpp
void KGEN::buildCheckLITPipeline(mlir::PassManager &pm,
                                 const CompilationOptions &options) {
  // Lower semantic control flow operations like lit.return to terminators and
  // diagnose unreachable code.
  pm.addPass(createLowerSemanticCF());
#ifndef MODULAR_PRODUCTION
  pm.addPass(createVerifyParameters());
#endif
  // Insert calls to destructors, reject use before free, and borrow check.
  pm.addPass(createCheckLifetimes(
      {/*extendTrivialDebugLifetimes=*/options.optimizationLevel == 0}));
}
```

User-facing diagnostics this adds on top of parse/type-check errors:

- `LowerSemanticCF` → unreachable-code diagnostics (e.g. code after `return`).
- `VerifyParameters` → parameter verification (non-production; internal).
- `CheckLifetimes` → Mojo's **borrow checker**: use-before-free,
  use-before-init, exclusivity/borrow violations, destructor-insertion checks.

#### How the diagnostics route back

```text
check pass emits an MLIR Diagnostic
  → mlir::SourceMgrDiagnosticHandler (on the clone's MLIRContext)
       converts to an llvm::SMDiagnostic, reports via the llvm::SourceMgr
  → llvm::SourceMgr::PrintMessage sees an installed DiagHandler and routes to it
  → that DiagHandler is the handlerCtx/handlerFn parseDocument installed (Sink 2)
  → appended to handlerCtx.smDiagnostics → published as publishDiagnostics
```

The enabling detail: `checkModuleSemantics` builds its
`mlir::SourceMgrDiagnosticHandler` over the **document's `sourceMgr`** — the
same one whose `DiagHandler` `parseDocument` installed before calling
`parseDocumentImpl` (and `checkModuleSemantics` runs *inside*
`parseDocumentImpl`, while that handler is still active). So pipeline
diagnostics land in the very same `smDiagnostics` collection as parse
diagnostics, indistinguishable at publish time.

#### What does *not* feed back

- **The cloned module is discarded** — a local `OwningOpRef` destroyed on
  return. This is deliberate: the check passes *mutate* IR (`LowerSemanticCF`
  lowers ops, `CheckLifetimes` inserts destructor calls), and that must not
  touch the pristine parser IR backing the `SymbolIndex`/hover/definition.
- **The `SymbolIndex` is untouched** — this is a `PassManager` run, not a parse;
  no listener fires.
- **The `LogicalResult` is essentially ignored** — failure just logs a debug
  line; only the emitted diagnostics reach the client.

#### Connections

- The `hasParserErrors` guard is set by Sink 2 (a `DK_Error` flips it), so
  semantic checking only runs on a syntactically/type-valid parse — no
  borrow-check noise on top of parse failures.
- This is *why* `cloneDeclModuleForCompilation` marks empty-body `FnOp`s
  `external` (the downstream consequence noted under `parseFileForLSP`): the LSP
  parse leaves imported bytecode functions as 0-block stubs, and the check
  pipeline would choke on them, so the clone patches them first.
  - The guard and the patch are *orthogonal*, not contradictory. "Valid parse"
    (`hasParserErrors == false`) is about the **user's source file**; empty
    bodies are about **imported** functions left at signature-only depth by
    `resolveSignaturesForLSP`. The user's own functions are body-resolved by
    `resolveForLSP`; if that failed it would emit a `DK_Error`, flip
    `hasParserErrors`, and skip this step — so when `checkModuleSemantics` runs,
    only *imported* stubs are empty-bodied. Marking them `external` (a function
    with no body must be `external`) is both verifier-legal and semantically
    correct ("defined elsewhere"); the passes only borrow-check the user's code.
- Limitation: these borrow-check/lifetime errors are still published with
  `source = "mojo"` / `category = "Parse Error"`; the LSP does not distinguish
  parse vs type-check vs borrow-check errors.

## Specialized parses: answering cursor queries

A specialized parse answers one cursor-positioned request and throws everything
away. It reuses the generalized parse's routine (`parseFileForLSP`) but swaps
the listener, the location filter, the sink, and the context lifetime.

### Common shape

All specialized parses (`codeComplete`, `signatureHelp`) share this shape, set
up by `parseCompletionImpl` in `Mojo/lib/MojoTooling/CodeComplete.cpp`:

1. **Throwaway context.** A fresh `SourceMgr` + `MLIRContext` +
   `MojoParserContext`, independent of the document's persistent `Context`,
   `SourceMgr`, and `SymbolIndex`. This is the "re-parses on every keystroke"
   cost.
2. **Diagnostics suppressed.** A no-op `SourceMgr` diag handler is installed —
   specialized parses never report errors (no Sink 2).
3. **Feature-specific listener** installed via `parserConfig.parserListener`.
4. **Location filter = the cursor.** `computeCompletionRange` converts the
   cursor byte offset into an `SMRange`; the listener's `isInterestedInLoc`
   accepts only matches in/around it.
5. **Underlying parse = `parseFileForLSP`** (the default parser callback).
6. **Transient sink:** results are collected into a buffer and returned to the
   client; nothing persists.

### `codeComplete`

Files: `Mojo/lib/MojoTooling/CodeComplete.cpp`,
`MojoTextDocument::onCodeCompletionSyncImpl` in
`KGEN/tools/mojo-lsp-server/MojoServer.cpp`.

Key idea: **completion is not a special parse mode.** It is the same
`parseFileForLSP` parse, run with a different `ParserListener`
(`CodeCompletionListener`) plus a location filter. The parser is completely
completion-agnostic; the listener decides what to harvest at the cursor.

#### The entry point

`MojoTextDocument::onCodeCompletionSyncImpl` reduces the cursor to a byte offset
and calls the **static** `MojoParserContext::codeComplete` with a **fresh
`MLIRContext`**:

```1993:2002:KGEN/tools/mojo-lsp-server/MojoServer.cpp
  llvm::SourceMgr &sourceMgr = getSourceMgr();
  const llvm::MemoryBuffer *buffer =
      sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());

  // Query the mojo parser for potential completion results.
  uint64_t rawCompletePos =
      completeLoc.getPointer() - buffer->getBuffer().data();
  MLIRContext mlirContext{MLIRContext::Threading::DISABLED};
  return MojoParserContext::codeComplete(*buffer, rawCompletePos, &mlirContext,
                                         getCompilationOptions());
```

This does **not** reuse the document's persistent `MojoParserContext`,
`SourceMgr`, or `SymbolIndex`. Each completion request is an independent,
throwaway parse — the source of the "re-parses on every keystroke" cost.

#### Setup (`parseCompletionImpl`)

1. Add the buffer to a fresh `SourceMgr`.
2. **Suppress diagnostics** (install a no-op diag handler) — completion never
   reports errors.
3. Build a `ParserConfig` with `parserListener = &listener`.
4. Create a fresh `MojoParserContext`.
5. `computeCompletionRange(...)` — convert the byte offset into an `SMRange`
   bracketing the partial identifier under the cursor: `Start` walks back over
   identifier characters (so `pri|nt` looks up from the start of `pri`), `End`
   is the next token boundary (via the parser lexer's `findStartOfNextToken`).
6. Run the parse via the callback -> `parseFileForLSP(bufId)`.

#### The mechanism: location-filtered listener hooks

The parser fires listener callbacks through `SharedState`, each gated by
`isInterestedInLoc`. `CodeCompletionListener` returns true **only inside the
completion range**:

```96:98:Mojo/lib/MojoTooling/CodeComplete.cpp
  bool isInterestedInLoc(SMLoc parserLoc) override {
    return containsLoc(completionRange, parserLoc);
  }
```

So the parser resolves the whole file but offers candidates only at the cursor.
The two hooks that matter (distinct from the `on*Decl`/`onRef` hooks the symbol
index uses):

- `onMemberLookup` — member access (`foo.ba|`) or bare-name lookup (`ba|`, with
  `searchParentScopes=true`).
- `onImport` (both overloads) — `import` / `from X import` at the cursor.

#### On-demand resolution at the cursor

When the listener is interested, the parser lazily resolves only the target decl
plus its immediate children (`resolveDeclForListenerLookup` ->
`resolveBody(decl)` then `resolveBody` on each child). This dovetails with
`parseFileForLSP`'s "signatures-only for externals": the type you complete on is
body-resolved only when you trigger the lookup.

#### Turning decls into results

`CodeCompletionListener::onMemberLookup` maps each child `ASTDecl` to a typed
`CodeCompletionResult` (module/package/struct/trait/function/field/variable) via
a `TypeSwitch`, attaching markdown docs from `getFullMarkdownString`. The
`onImport` path instead scans the module search directories on the filesystem,
spinning up throwaway contexts with `parseFileOrPackageNonRecursive` just to
pull each candidate module's doc string.

#### What `codeComplete` uses from the parser

| Parser facility                                                          | Used for                                        |
|--------------------------------------------------------------------------|-------------------------------------------------|
| `parseFileForLSP`                                                        | the underlying parse (same lazy entry point)    |
| `ParserListener::isInterestedInLoc`                                      | location filter — fire hooks only at the cursor |
| `ParserListener::onMemberLookup`                                         | member / identifier candidates                  |
| `ParserListener::onImport`                                               | import / module candidates                      |
| `resolveDeclForListenerLookup` (`resolveBody`)                           | on-demand resolution of target + children       |
| `findStartOfNextToken` + lexer                                           | compute completion range from the byte offset   |
| `MojoASTDeclRef` / `PublicDecl` (`getChildren`, `getFullMarkdownString`) | result kinds + docs                             |
| `getModuleSearchDirectories` + `parseFileOrPackageNonRecursive`          | filesystem import results + docs                |

#### Contrast with the generalized parse

| Aspect                      | Generalized parse                    | `codeComplete`                        |
|-----------------------------|--------------------------------------|---------------------------------------|
| `MojoParserContext`         | document's persistent one            | **fresh, throwaway** per request      |
| `SourceMgr` / `MLIRContext` | document's                           | **fresh** per request                 |
| Listener                    | `LSPParserListener` (symbol index)   | `CodeCompletionListener` (candidates) |
| Hooks consumed              | `on*Decl`, `onRef`, `onModuleImport` | `onMemberLookup`, `onImport`          |
| `isInterestedInLoc`         | "in main file"                       | "in completion range" (the cursor)    |
| Diagnostics                 | collected -> published               | **suppressed**                        |
| Output                      | `SymbolIndex` + diagnostics          | `vector<CodeCompletionResult>`        |
| Reuses prior parse?         | yes (cached on the document)         | **no** — full re-parse every time     |

### `signatureHelp`

Files: `Mojo/lib/MojoTooling/CodeComplete.cpp`,
`MojoTextDocument::onSignatureHelpSyncImpl` /
`MojoDocument::onSignatureHelpSync` in
`KGEN/tools/mojo-lsp-server/MojoServer.cpp`.

`signatureHelp` is the **twin of `codeComplete`** — same throwaway-context,
`parseFileForLSP`-based, diagnostics-suppressed parse dispatched through the
same `parseCompletionImpl`. The only differences are which hooks the listener
consumes and what it builds.

#### Hooks: call resolution, not member lookup

Where completion uses `onMemberLookup`/`onImport`, signature help uses the two
**call-resolution** hooks, reflecting that a Mojo callable has two operand
lists:

- `onCall` — runtime argument lists: `f(a, b|)`.
- `onParameterBinding` — compile-time parameter lists: `Foo[a, b|]`.

The location filter is coarser than completion's: it filters to the whole
completion **buffer**, then refines per-operand inside the hook.

```249:253:Mojo/lib/MojoTooling/CodeComplete.cpp
  bool isInterestedInLoc(SMLoc loc) override {
    // Filter at a high level for locations in the completion buffer, we'll
    // filter further when examining calls.
    return completionBufferId == sourceMgr.FindBufferContainingLoc(loc);
  }
```

#### Finding the active parameter

`onCall` walks the call's operands and picks the one whose range contains the
cursor; that index becomes `activeParameter` (what the editor bolds). If the
cursor is at the trailing `)` with no kwargs, it points at the next,
not-yet-typed argument.

#### Building signatures (one per overload)

Both hooks receive `ArrayRef<ASTDecl *> decls` — multiple candidates due to
overloading — and emit one `Signature` per viable overload. Two notable rules:

- Overload filtering: skip overloads that can't accept this many operands
  (`operands.size() > fnDecl->getArguments().size()`).
- `self` adjustment: for methods, bump `activeParameter` past the implicit
  `self` so the highlight lines up with the user-visible argument.

The label text comes from the parser's `getDeclarationSnippet`, which also
returns per-argument offsets (highlight spans into the label string).
`addDeclDocAndParametersToSignature` zips arguments with offsets to attach each
parameter's span + markdown doc. `onParameterBinding` is the same shape but
handles both `PublicFunctionDecl` and `PublicStructDecl` (structs are
parameterized) via `getParameters()`.

#### Result shape and LSP mapping

The listener fills a `SignatureHelpResult`
(`Mojo/include/Mojo/MojoTooling/ CodeComplete.h`): a list of
`Signature{label, documentation, parameters}` plus `activeSignature` /
`activeParameter`. Each `Parameter` is a `labelOffset` (highlight span) +
markdown `documentation`. The LSP side maps this to `lsp::SignatureHelp` and
then repackages into the extended `lsp::SignatureHelp2` (wrapping docs as
`MarkupContent{Markdown, ...}`). The `labelOffset` is what lets the editor
underline the active argument within the rendered signature.

#### What `signatureHelp` uses from the parser

| Parser facility                                                                         | Used for                                  |
|-----------------------------------------------------------------------------------------|-------------------------------------------|
| `parseFileForLSP` (via `parseCompletionImpl`)                                           | the underlying parse (same as completion) |
| `ParserListener::isInterestedInLoc`                                                     | coarse buffer filter, refined per-call    |
| `ParserListener::onCall`                                                                | runtime argument help `f(a, b\|)`         |
| `ParserListener::onParameterBinding`                                                    | compile-time parameter help `Foo[a, b\|]` |
| `PublicFunctionDecl` / `PublicStructDecl` (`getArguments`, `getParameters`, `isMethod`) | overload candidates + self detection      |
| `getDeclarationSnippet`                                                                 | signature label + per-arg offsets         |
| `getFullMarkdownString` / `getMarkdownDocString`                                        | signature + per-parameter docs            |

#### Contrast: `codeComplete` vs `signatureHelp`

| Aspect               | `codeComplete`                                     | `signatureHelp`                               |
|----------------------|----------------------------------------------------|-----------------------------------------------|
| Listener             | `CodeCompletionListener`                           | `SignatureHelpListener`                       |
| Hooks consumed       | `onMemberLookup`, `onImport`                       | `onCall`, `onParameterBinding`                |
| `isInterestedInLoc`  | tight completion range                             | whole buffer, refined per-operand             |
| On-demand resolution | `resolveDeclForListenerLookup` (target + children) | none extra; reads the call's resolved decls   |
| Output               | `vector<CodeCompletionResult>` (candidates)        | `SignatureHelpResult` (labels + active param) |
| Multiplicity         | one item per visible member                        | one signature per viable overload             |

Everything else (fresh throwaway context, suppressed diagnostics, full re-parse
per request, cursor-as-`isInterestedInLoc`-filter) is shared with
`codeComplete`.

### Docstring code blocks: `parseREPLExpression` (the exception)

Files: `Mojo/lib/MojoTooling/ParserDriverREPL.cpp`,
`MojoDocStrings::addDocString` / `processDocStrings` in
`KGEN/tools/mojo-lsp-server/MojoServer.cpp`.

A Mojo docstring can embed ` ```mojo ` code blocks. To offer
diagnostics/hover/completion inside them, the LSP parses each block via
`parseREPLExpression`. This is a specialized parse, but the **exception** to the
common shape: it runs during the *main* document parse (`processDocStrings` ←
`parseDocumentImpl`) on the document's **persistent** `MojoParserContext`, not a
throwaway one. It is the single most divergent — and most brittle — parser
interaction in the LSP.

#### Divergence 1: the user's code is textually rewritten before parsing

The block text is not parsed as-is. `wrapExpressionText` synthesizes a whole
Mojo program around it — an import preamble, a `struct __mojo_repl_context__` of
persistent variables, a wrapper `def`, an `__mojo_repl_expr_impl__` def, and the
user's body indented inside a nested
`@__parameter def __mojo_repl_expr_body__` — then adds *that* as a new buffer
and parses it.

The new buffer is added to the document's **single, shared `SourceMgr`** — the
same one the generalized parse uses (the persistent `MojoParserContext` is built
over `mainDoc.sourceMgr`, and `parseREPLExpression` calls `AddNewSourceBuffer`
on it). So after a parse the `SourceMgr` holds the main file
(`getMainFileID()`), imported files, *and* one synthetic wrapped buffer per code
block. This is exactly why `containsLocation`
(`FindBufferContainingLoc == getMainFileID`) reads a wrapped-buffer loc as "not
in main file" — keeping the scaffolding out of the `SymbolIndex`/diagnostics —
and why `translateParserLoc` must map such locs back into the main file's
docstring region (Divergence 3). These buffers accumulate for the document
version's lifetime and are discarded when an edit builds a fresh `MojoDocument`.

```456:471:Mojo/lib/MojoTooling/ParserDriverREPL.cpp
  // Splat out the main body code inside of a nested def. This will allow for us
  // to redefine previous variables transparently.
  exprOS << "  var __mojo_repl_expr_failed = True\n"
            "  @__parameter\n"
            "  def __mojo_repl_expr_body__() raises -> None:\n";
  for (StringRef code : mainBodyCode) {
    exprOS << "    ";
    emitAndMapCode(code);
  }
```

#### Divergence 2: a line-classification heuristic splits the code

`extractExpressionCode` scans line-by-line to decide what is module-top-level
(imports, decorator + `fn`/`struct` declarations, absorbing their bodies by
indentation) vs executable body — reimplementing a slice of "is this top-level?"
*outside* the real parser.

#### Divergence 3: two layers of location mapping

Because the parsed buffer differs from the source, every location round-trips
through two interval maps: (1) docstring **processed-offset → source offset**
(escape sequences make the processed string shorter; `addDocString` builds a
per-byte table via `Lexer::buildProcessedToSourceOffsets`, with delicate
exclusive-end handling), and (2) **wrapped-buffer offset → input-expression
offset** (`REPLLocMapper`/`ExprLocMapper`, consulted by
`MojoDocument::translateParserLoc` for non-main-buffer locs).

#### Divergence 4: cross-block state threading

Blocks are chained: each imports the previous block's decls, and variables are
detected and re-injected as `__mojo_repl_context__` fields for the next block,
emulating REPL state. The REPL module decls accumulate in the persistent
context.

```1872:1876:KGEN/tools/mojo-lsp-server/MojoServer.cpp
    auto [moduleDecl, exprFnDecl] = ctx.parseREPLExpression(
        listener, bufferId, contents, "__mojo_repl_lsp_main",
        persistentVariables, prevDecl,
        /*parseForLSP=*/true);
    prevDecl = codeBlock->decl = moduleDecl;
```

#### Convergence: same lazy resolution

For `parseForLSP=true` it uses the same signatures-only-for-externals strategy
as the main parse (`resolveForLSP` + `resolveSignaturesForLSP`). The non-LSP
path (real REPL/notebook/LLDB) instead does full `resolveAllReferencedFrom`
because it compiles and runs the wrapped code.

```797:805:Mojo/lib/MojoTooling/ParserDriverREPL.cpp
  if (parseForLSP) {
    resolveForLSP(*sharedState.declResolver, moduleDecl);
    if (!sharedState.diags.isErrorEmitted())
      resolveSignaturesForLSP(*sharedState.declResolver);
  } else {
    // Keep unparsed decls alive — later cells may reference them.
    sharedState.declResolver->resolveAllReferencedFrom(
        moduleDecl, /*eraseUnparsedDecls=*/false);
  }
```

#### Divergence 5: diagnostics remapped, error state wiped per block

A `REPLDiagnosticHandler` remaps diagnostics from the wrapped buffer back to the
input and forwards them via the listener; then `diags.clear()` blunt-resets the
shared error state so the next block still parses.

#### Brittleness sources

1. **Textual code generation** of exact Mojo syntax — a grammar change breaks
   all code blocks at once.
2. **The line-classification heuristic** (`extractExpressionCode`) reimplements
   parser knowledge; misclassifies multi-line/continuation/new-decl forms.
3. **Forced 4-space body indentation** — fragile against user indentation
   (note the defensive `+1` newline-mapping comment).
4. **Double, off-by-one-prone location mapping** with explicit exclusive-end
   special-casing; mis-maps silently misplace or drop diagnostics/hover.
5. **Escape-sequence reconstruction** via `buildProcessedToSourceOffsets`.
6. **Synthetic-name leakage** (`__mojo_repl_*`) filtered only by
   `isHiddenPersistentVariable`.
7. **Accumulating persistent state** across blocks; a mishandled failure (no
   `removeLastREPLExpression`) can corrupt later blocks.
8. **Scaffolding diagnostics** with no input mapping must be dropped; per-block
   `diags.clear()` is blunt and can mask or leak diagnostics.
9. **Stdlib coupling**: the preamble hard-imports `std.memory.UnsafePointer` and
   `std.python.python.Python`; a stdlib reorg breaks every block.
10. **Cost amplification**: runs inside the main parse, and
    `checkModuleSemantics` is invoked **per code block**
    (`MojoServer.cpp:1878`).

> Completion/signature help *inside* a code block (`CodeBlock::onCodeCompletion`
> / `onSignatureHelp`) likewise reuse the persistent context via
> `codeCompleteREPLExpression` / `signatureHelpREPLExpression` (which replay the
> accumulated REPL modules into a completion cache), not a throwaway parse.

## Reference: the full `ParserListener` interface

The `ParserListener` hooks (full interface in
`Mojo/include/Mojo/MojoParser/EntryPoint.h`, lines 148–233), annotated with
which persona overrides each. "Generalized" = the document parse's
`LSPParserListener`; "Completion"/"SigHelp" = the specialized listeners.

| Hook                                   | Fires when…                                     | Consumed by                                |
|----------------------------------------|-------------------------------------------------|--------------------------------------------|
| `isInterestedInLoc`                    | parser asks "do you care about this loc?"       | all (different filter each)                |
| `onAliasDecl`                          | an `alias` is resolved                          | Generalized -> index symbol                |
| `onArgumentDecl`                       | a function argument decl resolved               | Generalized -> index symbol                |
| `onFunctionDecl`                       | a `def`/`fn` (incl. methods, closures) resolved | Generalized -> index symbol                |
| `onModuleDecl`                         | a `module` decl created                         | Generalized (skips main file's own module) |
| `onModuleImport`                       | `from M import …` resolved                      | Generalized -> register a reference        |
| `onParameterDecl`                      | a function/struct parameter resolved            | Generalized -> index symbol                |
| `onStructDecl`                         | a `struct` resolved                             | Generalized -> index symbol                |
| `onStructFieldDecl`                    | a struct field resolved                         | Generalized -> index symbol                |
| `onTraitDecl`                          | a `trait` resolved                              | Generalized -> index symbol                |
| `onVariableDecl`                       | a `var`/`let` resolved                          | Generalized -> index symbol                |
| `onRef`                                | a reference's decls are known                   | Generalized -> register a reference        |
| `onImport(loc)` / `onImport(pkg, loc)` | an import is being resolved                     | Completion (no-op in Generalized)          |
| `onMemberLookup`                       | a member is being looked up                     | Completion (no-op in Generalized)          |
| `onCall`                               | a call is being resolved with operands          | SigHelp (no-op in Generalized)             |
| `onParameterBinding`                   | parameters are being bound                      | SigHelp (no-op in Generalized)             |

The base-class defaults are all no-ops (`Mojo/lib/MojoParser/EntryPoint.cpp`,
~609–633), so any hook a given listener does not override simply does nothing
for that parse. This is what makes the persona hook sets disjoint.

## Threading: using a single-threaded parser safely

The Mojo parser's resolution is inherently single-threaded, but the LSP server
is multi-threaded. The server's invariant is simple:
**never let two threads touch the same parser state.** It enforces this on three
axes.

### The threads

| Thread                 | Role                                                                          |
|------------------------|-------------------------------------------------------------------------------|
| Main thread            | JSON-RPC I/O loop (`transport.run`); receives/dispatches, returns immediately |
| Debouncer worker       | one `std::thread` that coalesces `didChange` edits and fires the parse        |
| `MLRT::CPUDevice` pool | worker threads that execute the async tasks (parse + queries)                 |

### Axis 1 — across documents: isolated parser state

Each document's `Context` builds a fresh `MLIRContext` **with threading
disabled** and its own `MojoParserContext` / `SourceMgr` / `SymbolIndex`:

```728:740:KGEN/tools/mojo-lsp-server/MojoServer.cpp
struct MojoDocument::Context {
  Context(MojoDocument &mainDoc)
      : mlirContext(MLIRContext::Threading::DISABLED),
        ...
    parserContext =
        std::make_unique<MojoParserContext>(mainDoc.sourceMgr, parserConfig);
```

So MLIR never multithreads inside a parse, and two *different* documents can be
parsed concurrently on different pool threads with zero shared mutable state.
The only cross-document structure is the `files` map, guarded by `filesMutex`.

### Axis 2 — within a document: a serial FIFO task chain

Every operation touching a document's parse state (the parse itself *and* every
feature query) is dispatched via `startTask`, which chains tasks so they run
one-at-a-time, lock-free:

```329:339:KGEN/tools/mojo-lsp-server/MojoDocument.h
  template <typename FnT>
  void startTask(FnT &&fn) {
    auto [previous, current] = enqueueNewTask();

    previous.andThenAsync([doc = RCRef<MojoDocument>::copy(this),
                           fn = std::forward<FnT>(fn),
                           current = std::move(current)]() mutable {
      def(*doc);
      std::move(current).emplace();
    });
  }
```

`enqueueNewTask` (under `currentTaskMutex`) bumps `chainIndex`, captures the
`previous` chain to wait on, and installs a fresh `current` chain for the next
task. Each task waits on `previous`, runs `fn`, then readies `current`. The
mutex guards only the brief pointer swap — never the parser work — so each task
gets **exclusive, lock-free access** to the document's parse state for its whole
duration.

Consequences:

- A query arriving mid-parse transparently **waits** behind the parse, then sees
  a fully built index.
- `startDocumentParse` asserts `chainIndex == 0`: the parse is always the
  *first* task on a fresh document, so every query necessarily chains after it.
- Each task captures an `RCRef<MojoDocument>`, keeping the document alive while
  in flight even after it leaves the `files` map.

### Axis 3 — across versions: fresh document per edit + invalidation

The server never mutates a live parse in place. On every (debounced) edit,
`addDocumentImmediate` invalidates the existing document and replaces it with a
brand-new one, whose chain restarts at index 0:

```2386:2400:KGEN/tools/mojo-lsp-server/MojoServer.cpp
    // If a document already exists, invalidate that version.
    MLRT::CPUDevice &cpuDevice = *ctx->get<MLRT::CPUDevice>();
    if (it->second) {
      it->second->invalidate();
    }
    ...
    it->second->startDocumentParse(telemetryCtx, progressMgr);
```

`invalidate()` flips an atomic flag; every task lambda checks `isInvalidated` at
the top and bails with `ContentModified` (`replyOutdatedRequest`) rather than
computing against stale state. The superseded document object survives only
until its in-flight tasks drain (via the captured `RCRef`s). So there is never
in-place mutation of a live parse, and stale work is abandoned promptly.

### Debounce, generations, modes, shutdown

- **Debounce:** `didChange` stores the new contents and calls
  `debouncer->scheduleUpdate(...)`; the worker waits ~150 ms after the last
  keystroke before creating the new document — coalescing bursts into one parse.
- **Generation counter (TOCTOU):** each open bumps a generation; the debounced
  callback re-checks it before adding, so a close+reopen can't apply a stale
  parse. `didClose` cancels pending updates and erases the generation.
- **Single-threaded mode:** the `CPUDevice` is built
  `withSingleThreaded(singleThreaded).withMainWillNotDonate()`; `--mojo-test`
  forces single-threaded for deterministic output. Per-document serialization is
  identical either way.
- **Shutdown:** flush/destroy the debouncer (joining its worker), then
  invalidate each document and `await` its task chain before clearing the
  `files` map.

## Follow ups

Out of scope (read side): how feature handlers query the `SymbolIndex` to answer
hover/definition/references/rename/doc-symbols/semantic-tokens. These reads do
not invoke the parser, so they are outside this doc's focus.
