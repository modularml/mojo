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
// This file defines the preferred diagnostics machinery used by the frontend.
// It is intentionally similar to the MLIR diagnostics machinery, but disallows
// printing MLIR types (forcing use of ASTTypes) and supports additional
// features like source ranges.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMPILER_DIAGS_H
#define SUPPORT_COMPILER_DIAGS_H

#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace llvm {

template <>
struct DenseMapInfo<SMFixIt> {
  using KeyTy = SMFixIt;

  static unsigned getHashValue(const KeyTy &k) {
    SMRange range = k.getRange();
    return DenseMapInfo<std::tuple<const char *, const char *, StringRef>>::
        getHashValue(
            {range.Start.getPointer(), range.End.getPointer(), k.getText()});
  }

  static bool isEqual(const KeyTy &a, const KeyTy &b) {
    SMRange aRange = a.getRange();
    SMRange bRange = b.getRange();
    return aRange.Start == bRange.Start && aRange.End == bRange.End &&
           a.getText() == b.getText();
  }
};
} // namespace llvm

namespace M {
using llvm::SMFixIt;
using llvm::SMLoc;
using llvm::SourceMgr;
class InflightDiag;
class SourceRange;
class FixIt;

//===----------------------------------------------------------------------===//
// AutoFixItHandler
//===----------------------------------------------------------------------===//

/// This class is used to collect and apply fix-its automatically.
/// It supports two output modes:
/// - Apply mode (default): Directly modifies source files with fix-its
/// - Export mode: Exports fix-its to a YAML file in clang-tidy format
///   The exported YAML is compatible with 'clang-apply-replacements'.
class AutoFixItHandler {
public:
  /// Create a handler that applies fix-its directly to files.
  AutoFixItHandler() = default;

  /// Create a handler that exports fix-its to a YAML file instead of applying.
  explicit AutoFixItHandler(StringRef exportPath);

  ~AutoFixItHandler() = default;

  /// Register a fix-it to be applied later.
  void registerFixIt(const llvm::MemoryBuffer *buffer, llvm::SMFixIt fixIt);

  /// Apply all registered fix-its directly to files.
  /// Only valid when not in export mode.
  void applyFixIts();

  /// Export fix-its to a YAML file in clang-tidy format.
  /// Returns success() if the file was written successfully, or an error
  /// if writing failed. Only valid when in export mode.
  ErrorOrSuccess exportFixIts();

  /// Return true if there are any fix-its to apply.
  bool hasFixIts() const { return !fixits.empty(); }

  /// Return true if this handler is in apply mode.
  bool isApplyMode() const { return !exportPath.has_value(); }

private:
  /// A map of buffers (i.e. source files) to a vector of fix-its to apply to
  /// that buffer. Uses MapVector for deterministic file ordering (by insertion
  /// order). Fix-its within each file are deduplicated on insert and sorted by
  /// offset before export for deterministic output.
  llvm::MapVector<const llvm::MemoryBuffer *, llvm::SmallVector<llvm::SMFixIt>>
      fixits;

  /// Optional path to output YAML file. If set, handler is in export mode.
  std::optional<std::string> exportPath;
};

//===----------------------------------------------------------------------===//
// Diags
//===----------------------------------------------------------------------===//

class Diags {
public:
  Diags(SourceMgr &sourceMgr, MLIRContext *context, bool useMLIRDiagnostics,
        int maxNotesPerDiagnostic, StringRef stripFilenamePrefix,
        bool disableWarnings, bool warningsAsErrors,
        void *extraContext = nullptr,
        AutoFixItHandler *autoFixItHandler = nullptr);
  ~Diags();

  llvm::SourceMgr &sourceMgr;
  MLIRContext *const context;

  /// This is an optional client specific context token that can be used by
  /// clients that need additional data to emit diagnostics.
  void *const extraContext;

  /// Return the identifier for the main buffer in the SourceMgr.
  StringAttr getBufferNameIdentifier() const;

  bool isErrorEmitted() const { return errorEmitted; }

  bool isDiagnosticEmitted() const { return diagnosticEmitted; }

  /// Clear out the current diagnostic state.
  void clear() { errorEmitted = diagnosticEmitted = false; }

  /// Emit an error.
  InflightDiag emitError(Location loc, const Twine &message);
  InflightDiag emitError(llvm::SMLoc loc, const Twine &message);

  /// Emit a warning.
  InflightDiag emitWarning(Location loc, const Twine &message);
  InflightDiag emitWarning(llvm::SMLoc loc, const Twine &message);

  /// Encode the specified source location information into a Location object
  /// for attachment to the IR or error reporting.  This always returns a
  /// FileLineColLoc.
  FileLineColLoc translateLocation(llvm::SMLoc loc) const;

  /// Canonicalize a filename by stripping `stripFilenamePrefix` prefix from it
  /// if applicable.
  std::string getCanonicalFilename(StringRef filenameRef) const;

  /// Decode the specific MLIR location information into an SMLoc for use with
  /// the SourceMgr. This returns an invalid SMLoc if the location is not
  /// understood.
  SMLoc convertLocToSMLoc(LocationAttr loc) const;

  /// If `loc` refers to a source file, remember `includeLoc` as that file's
  /// "Included from" root for when the file is later rendered into the
  /// SourceMgr to display a diagnostic. This is for the source files of a
  /// precompiled package, which are never opened during parsing (only at
  /// diagnostic time), so their buffers would otherwise have no include
  /// location. First-wins, and a no-op once the file's buffer already exists.
  void recordImportedFileIncludeLoc(LocationAttr loc, SMLoc includeLoc);

  /// Convert the given source range to an SMRange.
  llvm::SMRange convertToSMRange(SourceRange range) const;

  /// This is a helper object that allows turning Location objects into SMLoc's.
  class SourceMgrLocationMapper;
  std::unique_ptr<SourceMgrLocationMapper> sourceMgrMapper;

  /// Specify a function used to adjust the end-point of a token given a pointer
  /// to the start of the token.
  void setTokenEndPointAdjustmentFn(std::function<void(SMLoc &)> fn) {
    tokenEndPointAdjustmentFn = std::move(fn);
  }

  std::function<void(SMLoc &)> tokenEndPointAdjustmentFn;

  /// This is true if we should use MLIR for diagnostics (e.g. to enable
  /// -verify-diagnostics and other MLIR testing features), but we prefer
  /// llvm::SourceMgr for better QoI: it supports source ranges and FixIt hints.
  const bool useMLIRDiagnostics;

  /// When non-null, this handler collects and applies fix-its automatically.
  AutoFixItHandler *autoFixItHandler;

private:
  /// A set of diagnostic handler routines that adjusts locations using
  /// `getCanonicalFilename`.
  void printImportStack(SMLoc IncludeLoc, raw_ostream &OS) const;
  void emitDiagnostic(const llvm::SMDiagnostic &diag) const;

  friend class InflightDiag;
  Diags(const Diags &) = delete;

  /// This is the StringAttr for the main buffer identifier. It is type erased
  /// to void* to reduce header pollution. This field is lazy initialized to
  /// handle the case where the main buffer is added after the Diags object is
  /// constructed.
  mutable std::optional<const void *> bufferNameIdentifier;

  /// This is set to true if an error occurred at any point processing the
  /// file.
  bool errorEmitted = false;

  /// This is set to true if any diagnostic occurred at any point processing the
  /// file.
  bool diagnosticEmitted = false;

  /// Configuration for how many notes to print for a diagnostic.
  int maxNotesPerDiagnostic;

  bool disableWarnings;
  bool warningsAsErrors = false;
};

/// This class represents a diagnostic that is built up by the parser and
/// emitted when destroyed.  This allows clients to incrementally build up the
/// message, attach notes, ranges and fixit hints all with a simple interface.
///
/// Each diagnostic is made up of a primary message and is optionally followed
/// by any number of note messages.
class InflightDiag {
public:
  InflightDiag(Location loc, Diags &diags, bool isWarning);
  ~InflightDiag();
  InflightDiag(InflightDiag &&other);
  InflightDiag &operator=(InflightDiag &&other);

  // This class in non-copyable.
  InflightDiag(const InflightDiag &) = delete;
  InflightDiag &operator=(const InflightDiag &) = delete;

  /// Abandon emission of this message, this will make it be a noop when its
  /// destructor runs.
  void abandon() { diags = nullptr; }

  // InflightDiag always converts to failure.  This allows certain patterns to
  // be more ergonomic.
  operator LogicalResult() const { return failure(); }
  operator ParseResult() const { return failure(); }

  /// Add a note to this diagnostic at the specified location, and change the
  /// emission point to start filling it in.
  InflightDiag attachNote(Location loc) &&;
  InflightDiag &attachNote(Location loc) &;
  InflightDiag attachNote(SMLoc loc) &&;
  InflightDiag &attachNote(SMLoc loc) &;

  /// Get the location of the most recent message in the diagnostic.
  Location getPrimaryLoc() const;
  Location getLastLoc() const;

  // Insertion operations for various things that contribute to the current
  // messages.  These are implemented with addToDiagnostic methods.
  template <typename Arg>
  InflightDiag &operator<<(Arg &&value) & {
    addToDiagnostic(std::forward<Arg>(value), *this);
    return *this;
  }
  template <typename Arg>
  InflightDiag operator<<(Arg value) && {
    addToDiagnostic(std::forward<Arg>(value), *this);
    return std::move(*this);
  }

  /// These methods can be used by addToDiagnostic impls to add things to the
  /// diagnostic.
  void addText(const Twine &text);
  void addCustomLineText(const Twine &text);
  void addSourceRange(SourceRange range);
  void addFixIt(FixIt fixIt);
  void addDiag(InflightDiag &&otherDiag);

  /// Return the diagnostic object if this is attached.
  Diags *getDiags() const { return diags; }

private:
  void emitMLIRDiagnostic();
  void emitSourceMgrDiagnostic();
  void emitAutoFixItDiagnostic();

  /// Each message in a diagnostic must have a location and text, and may
  /// have any number of highlighted ranges and fixit hints.
  struct Message;

  /// Format a diagnostic message into an llvm::SMDiagnostic ready for printing.
  /// Note; the diagnostic text is provided as `text`, and not taken from
  /// `message.text`.
  llvm::SMDiagnostic getAsSMDiagnostic(const Message &message,
                                       const std::string &text,
                                       SourceMgr::DiagKind kind,
                                       const std::string &customLineStr = "");

  // We store the primary diagnostic and any notes in this vector.  The primary
  // diagnostic is always first, and always present.  This uses a std::vector
  // so Message can be defined out of line, and to make the move operation
  // efficient.
  std::vector<Message> messages;

  /// This is the diagnostic object to emit to, or null if abandoned.
  Diags *diags;

  /// True if this is a warning, false if this is an error.
  bool isWarning;
};

/// Represents a range in source code.  The default use-case for this class is
/// to represent source ranges in terms of lit token positions beginnings, where
/// the start/end of the range indicate the beginning of the tokens included in
/// the range.  This makes it easier to construct and work with.
///
/// For example, in the expression `yoda + 492`, a range with the start/end both
/// pointing to the 'y', would indicate the full identifier "yoda".  Similarly,
/// a range pointing to the 'y' and the '4' would cover the entire span from the
/// start of y through the end of 2.
///
/// There are some narrow cases where you may want to diagnose within a token,
/// e.g. complaining about a format character in a string literal.  In those
/// cases, you may use a 'byte-level' string, which uses a half-open range and
/// is not extended to include the end of the token.
class SourceRange {
public:
  /// Build a null range.
  SourceRange() = default;

  /// Build a normal token-start range.
  SourceRange(SMLoc start, SMLoc end);
  SourceRange(llvm::SMRange range);

  /// Build a byte-level range.
  static SourceRange getByteLevel(SMLoc start, SMLoc end);

  SMLoc getStart() const;
  SMLoc getEnd() const;
  bool isValid() const { return start != nullptr; }
  bool isByteLevel() const { return byteLevel; }

  /// Return an SMRange that corresponds to this source range.
  llvm::SMRange getSMRange() const;

private:
  const char *start = nullptr, *end = nullptr;
  bool byteLevel = false;
};

/// A FixIt hint is a source rewrite that some IDEs can apply automatically when
/// errors occur.  Generation of FixIt hints is great for QoI.  Error recovery
/// in the parser must always follow the logic that would have happened if the
/// FixIt was applied so the user doesn't get a different downstream error after
/// applying the FixIt hint.
class FixIt {
public:
  FixIt(SourceRange range, const Twine &replacement);

  /// This constructor creates a fixit that removes the specified token.
  static FixIt remove(SMLoc loc);

  /// This constructor creates a fixit that removes the specified token range.
  static FixIt remove(SourceRange range);

  /// This constructor creates a fixit that replaces the one token at the
  /// specified location with some text.
  static FixIt replaceToken(SMLoc loc, const Twine &text);

  /// This constructor creates a fixit that inserts some text before the token
  /// at the specified location, without replacing the token.
  static FixIt insertBeforeToken(SMLoc loc, const Twine &text);

  /// This constructor creates a fixit that inserts some text after the token
  /// at the specified location.
  static FixIt insertAfterToken(SMLoc loc, const Twine &text, Diags &diags);

  /// This is the source range to remove.
  SourceRange range;
  /// This is what to replace it with.
  std::string replacement;
};

// These methods enable adding common types to the current diagnostic.
void addToDiagnostic(const Twine &text, InflightDiag &diag);
void addToDiagnostic(char text, InflightDiag &diag);
void addToDiagnostic(size_t number, InflightDiag &diag);
void addToDiagnostic(StringAttr attr, InflightDiag &diag);

/// This adds a source range highlight.
void addToDiagnostic(SourceRange range, InflightDiag &diag);
/// This adds a fixit hint.
void addToDiagnostic(FixIt fixIt, InflightDiag &diag);
/// This concatenates another diagnostic.
void addToDiagnostic(InflightDiag &&otherDiag, InflightDiag &diag);

/// Compiler passes can emit warnings using MLIR diagnostics, and those
/// warnings are not seen by the Diags class above. Instantiating this
/// scoped handler in the driver (mojo-run, mojo-build, mojo-package) ensures
/// that warning handling works consistently across all warning emission
/// mechanisms.
class ScopedMLIRWarningHandler {
public:
  ScopedMLIRWarningHandler(MLIRContext *ctx, bool disableWarnings,
                           bool warningsAsErrors);

  /// Returns true if a warning was promoted to an error.
  bool wasErrorEmitted() const { return errorEmitted; }

private:
  // errorEmitted must be declared before handler so it's initialized before
  // handler's constructor captures its address.
  bool errorEmitted = false;
  mlir::ScopedDiagnosticHandler handler;
};
} // namespace M

#endif // SUPPORT_COMPILER_DIAGS_H
