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

#include "Support/Compiler/Diags.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/RewriteBuffer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

using llvm::SMRange;
using namespace M;

//===----------------------------------------------------------------------===//
// YAML Traits for clang-tidy compatible fix-it export
//===----------------------------------------------------------------------===//

namespace {
/// A single replacement in clang-tidy YAML format.
struct YAMLReplacement {
  std::string filePath;
  unsigned offset;
  unsigned length;
  std::string replacementText;
};

/// The top-level structure for clang-tidy YAML format.
struct YAMLTranslationUnitReplacements {
  std::string mainSourceFile;
  std::vector<YAMLReplacement> replacements;
};
} // namespace

namespace llvm::yaml {
template <>
struct MappingTraits<YAMLReplacement> {
  static void mapping(IO &io, YAMLReplacement &r) {
    io.mapRequired("FilePath", r.filePath);
    io.mapRequired("Offset", r.offset);
    io.mapRequired("Length", r.length);
    io.mapRequired("ReplacementText", r.replacementText);
  }
};

template <>
struct MappingTraits<YAMLTranslationUnitReplacements> {
  static void mapping(IO &io, YAMLTranslationUnitReplacements &doc) {
    io.mapRequired("MainSourceFile", doc.mainSourceFile);
    io.mapRequired("Replacements", doc.replacements);
  }
};
} // namespace llvm::yaml

LLVM_YAML_IS_SEQUENCE_VECTOR(YAMLReplacement)

//===----------------------------------------------------------------------===//
// SourceRange implementation
//===----------------------------------------------------------------------===//

SourceRange::SourceRange(SMLoc start, SMLoc end)
    : start(start.getPointer()), end(end.getPointer()) {
  assert(start.isValid() == end.isValid() &&
         "Start and End should either both be valid or both be invalid!");
}

SourceRange::SourceRange(SMRange range) : SourceRange(range.Start, range.End) {}

SourceRange SourceRange::getByteLevel(SMLoc start, SMLoc end) {
  auto result = SourceRange(start, end);
  result.byteLevel = true;
  return result;
}

SMLoc SourceRange::getStart() const { return SMLoc::getFromPointer(start); }
SMLoc SourceRange::getEnd() const { return SMLoc::getFromPointer(end); }

//===----------------------------------------------------------------------===//
// SourceMgrLocationMapper implementation
//===----------------------------------------------------------------------===//

/// This is a helper that allows us to translate mlir::Location objects to
/// llvm::SMLoc objects.
///
/// TODO(mlir upstream): This was refactored out of
/// SourceMgrDiagnosticHandlerImpl; upstream this.
class Diags::SourceMgrLocationMapper {
public:
  SourceMgrLocationMapper(MLIRContext *context, StringRef stripFilenamePrefix)
      : unknownBufferNameIdentifier(StringAttr::get(
            context, Diags::SourceMgrLocationMapper::kUnnamedFileSigil)),
        context(context), stripFilenamePrefix(stripFilenamePrefix) {}

  /// Constant string that we can use to signify an un-named file (usually means
  /// reading from stdin or something).
  static constexpr StringLiteral kUnnamedFileSigil = "<unknown>";
  /// StringAttr version of `kUnnamedFileSigil`.
  StringAttr unknownBufferNameIdentifier;

  /// Return the SrcManager buffer id for the specified file, or zero if none
  /// can be found.
  unsigned getBufferIDForFile(SourceMgr &sourceMgr, StringAttr filename);
  /// Get the name of the buffer so we can rapidly build Location objects on
  /// demand.
  StringAttr getFilenameFromBufferId(const SourceMgr &sourceMgr,
                                     unsigned bufferID);

  /// Convert a location to SMLoc.
  SMLoc convertLocToSMLoc(SourceMgr &sourceMgr, FileLineColLoc loc);
  SMLoc convertLocToSMLoc(SourceMgr &sourceMgr, Location loc);

  /// Convert a SMLoc to location.
  FileLineColLoc convertSMLocToLoc(SourceMgr &sourceMgr, SMLoc loc);

  /// Canonicalize a filename by stripping `stripFilenamePrefix` prefix from it
  /// if applicable.
  std::string getCanonicalFilename(StringRef filename) const;
  StringRef getStripFilenamePrefix() const { return stripFilenamePrefix; }

  /// Record the include location to stamp onto the filename's buffer if and
  /// when we resolve it. First-wins, and a no-op once the buffer is already
  /// resolved.
  void recordIncludeLoc(StringAttr filename, SMLoc includeLoc);

  /// Return the recorded "included from" location for `filename`, if any.
  /// Returns an invalid SMLoc if nothing was recorded.
  SMLoc getRecordedIncludeLoc(StringRef filename) const;

private:
  MLIRContext *context;
  std::string stripFilenamePrefix;
  /// What the mapper knows about a filename.
  struct FileEntry {
    /// The SourceMgr buffer for this file. Three states:
    /// - nullopt -> not yet looked up
    /// - 0 -> looked up, no buffer exists
    /// - N -> buffer N.
    std::optional<unsigned> bufId;
    /// Include location to stamp on the buffer if/when we resolve it.
    SMLoc includeLoc;
  };
  /// Maps known StringAttr filenames to what we know about them.
  llvm::DenseMap<StringAttr, FileEntry> filenameToBufId;
  /// Maps buffer IDs to their canonical filenames.
  llvm::DenseMap<unsigned, StringAttr> bufIdToCanonicalFilename;
};

unsigned
Diags::SourceMgrLocationMapper::getBufferIDForFile(llvm::SourceMgr &sourceMgr,
                                                   StringAttr filename) {
  FileEntry &entry = filenameToBufId[filename];
  // Check for an existing mapping to the buffer id for this file.
  if (entry.bufId)
    return *entry.bufId;

  // Look for a buffer in the manager that has this filename.
  for (unsigned i = 1, e = sourceMgr.getNumBuffers() + 1; i != e; ++i) {
    auto *buf = sourceMgr.getMemoryBuffer(i);
    if (buf->getBufferIdentifier() == filename.getValue())
      return *(entry.bufId = i);
  }

  // Otherwise, try to load the source file, resolving it to any recorded
  // include location.
  std::string ignored;
  return *(entry.bufId = sourceMgr.AddIncludeFile(filename.str(),
                                                  entry.includeLoc, ignored));
}

void Diags::SourceMgrLocationMapper::recordIncludeLoc(StringAttr filename,
                                                      SMLoc includeLoc) {
  if (!includeLoc.isValid())
    return;
  FileEntry &entry = filenameToBufId[filename];
  // First-wins, and only meaningful before the buffer is resolved: once we've
  // faulted the file in, its IncludeLoc is fixed.
  if (!entry.bufId && !entry.includeLoc.isValid())
    entry.includeLoc = includeLoc;
}

SMLoc Diags::SourceMgrLocationMapper::getRecordedIncludeLoc(
    StringRef filename) const {
  auto it = filenameToBufId.find(StringAttr::get(context, filename));
  if (it == filenameToBufId.end())
    return SMLoc();
  return it->second.includeLoc;
}

StringAttr Diags::SourceMgrLocationMapper::getFilenameFromBufferId(
    const SourceMgr &sourceMgr, unsigned bufferID) {
  // Lookup the cache first.
  auto bufferIt = bufIdToCanonicalFilename.find(bufferID);
  if (bufferIt != bufIdToCanonicalFilename.end())
    return bufferIt->second;

  if (sourceMgr.getNumBuffers() == 0)
    return unknownBufferNameIdentifier;

  auto mainBuffer = sourceMgr.getMemoryBuffer(bufferID);
  StringRef bufferName = mainBuffer->getBufferIdentifier();
  StringAttr filename;
  if (bufferName.empty()) {
    filename = unknownBufferNameIdentifier;
  } else {
    std::string relFilename = getCanonicalFilename(bufferName);
    filename = StringAttr::get(context, relFilename);
  }

  bufIdToCanonicalFilename[bufferID] = filename;
  // We already hold this buffer, so record it as resolved.
  filenameToBufId[filename].bufId = bufferID;
  return filename;
}

/// Get a memory buffer for the given file, or the main file of the source
/// manager if one doesn't exist. This always returns non-null.
SMLoc Diags::SourceMgrLocationMapper::convertLocToSMLoc(SourceMgr &sourceMgr,
                                                        FileLineColLoc loc) {
  // The column and line may be zero to represent unknown column and/or unknown
  /// line/column information.
  if (loc.getLine() == 0 || loc.getColumn() == 0)
    return SMLoc();

  // Default the buffer ID to 0 - this is the 'unknown' buffer ID since the
  // SourceMgr starts at 1. This will result in a default-constructed SMLoc
  // below unless we can match the filename in the loc.
  unsigned bufferId = 0;
  if (loc.getFilename() == kUnnamedFileSigil)
    bufferId = sourceMgr.getMainFileID();
  else
    bufferId = getBufferIDForFile(sourceMgr, loc.getFilename());

  if (!bufferId)
    return SMLoc();
  return sourceMgr.FindLocForLineAndColumn(bufferId, loc.getLine(),
                                           loc.getColumn());
}

SMLoc Diags::SourceMgrLocationMapper::convertLocToSMLoc(SourceMgr &sourceMgr,
                                                        Location loc) {
  if (auto fileLineCol = dyn_cast<FileLineColLoc>(loc))
    return convertLocToSMLoc(sourceMgr, fileLineCol);
  return SMLoc();
}

FileLineColLoc
Diags::SourceMgrLocationMapper::convertSMLocToLoc(SourceMgr &sourceMgr,
                                                  SMLoc loc) {
  unsigned bufferID = sourceMgr.FindBufferContainingLoc(loc);
  if (!bufferID)
    return FileLineColLoc::get(unknownBufferNameIdentifier, 0, 0);
  auto lineAndColumn = sourceMgr.getLineAndColumn(loc, bufferID);

  StringAttr bufferName = getFilenameFromBufferId(sourceMgr, bufferID);
  return FileLineColLoc::get(bufferName, lineAndColumn.first,
                             lineAndColumn.second);
}

std::string Diags::SourceMgrLocationMapper::getCanonicalFilename(
    StringRef filenameRef) const {
  if (!stripFilenamePrefix.empty() &&
      filenameRef.starts_with(stripFilenamePrefix))
    return filenameRef.drop_front(stripFilenamePrefix.size()).str();
  return std::string(filenameRef);
}

//===----------------------------------------------------------------------===//
// AutoFixItHandler
//===----------------------------------------------------------------------===//

AutoFixItHandler::AutoFixItHandler(StringRef exportPath)
    : exportPath(exportPath.str()) {}

/// Check if two SMFixIt objects are equal (same range and text).
static bool fixItEquals(const llvm::SMFixIt &a, const llvm::SMFixIt &b) {
  SMRange aRange = a.getRange();
  SMRange bRange = b.getRange();
  return aRange.Start == bRange.Start && aRange.End == bRange.End &&
         a.getText() == b.getText();
}

void AutoFixItHandler::registerFixIt(const llvm::MemoryBuffer *buffer,
                                     llvm::SMFixIt fixIt) {
  auto &bufferFixIts = fixits[buffer];
  // Deduplicate: only add if not already present.
  if (!llvm::any_of(bufferFixIts,
                    [&](const auto &f) { return fixItEquals(f, fixIt); }))
    bufferFixIts.push_back(fixIt);
}

void AutoFixItHandler::applyFixIts() {
  assert(isApplyMode() && "applyFixIts() called in export mode");

  for (auto &[buffer, bufferFixIts] : fixits) {
    llvm::RewriteBuffer rewriteBuffer;
    const char *bufferStart = buffer->getBufferStart();
    rewriteBuffer.Initialize(bufferStart, buffer->getBufferEnd());

    for (auto &fixIt : bufferFixIts) {
      SMRange fixItRange = fixIt.getRange();
      assert(fixItRange.isValid());
      const char *startLoc = fixItRange.Start.getPointer();
      const char *endLoc = fixItRange.End.getPointer();
      rewriteBuffer.ReplaceText(startLoc - bufferStart, endLoc - startLoc,
                                fixIt.getText());
    }

    std::string modified;
    llvm::raw_string_ostream resultStream(modified);
    rewriteBuffer.write(resultStream);

    // Write the modified buffer back to the file.
    llvm::Error err = llvm::writeToOutput(buffer->getBufferIdentifier(),
                                          [&](raw_ostream &os) {
                                            os << modified;
                                            return llvm::Error::success();
                                          });
    if (err)
      llvm::errs() << "Error writing buffer to file: " << err << "\n";
  }
}

ErrorOrSuccess AutoFixItHandler::exportFixIts() {
  assert(!isApplyMode() && "exportFixIts() called in apply mode");

  // Build the YAML document structure.
  YAMLTranslationUnitReplacements doc;
  doc.mainSourceFile = "";

  // MapVector provides deterministic file ordering (by insertion order).
  for (auto &[buffer, bufferFixIts] : fixits) {
    std::string filePath = buffer->getBufferIdentifier().str();
    const char *bufferStart = buffer->getBufferStart();

    // Sort fix-its by offset for deterministic output within each file.
    // This also makes the YAML easier to review manually.
    llvm::SmallVector<llvm::SMFixIt> sortedFixIts(bufferFixIts);
    llvm::sort(sortedFixIts, [](const llvm::SMFixIt &a,
                                const llvm::SMFixIt &b) {
      return a.getRange().Start.getPointer() < b.getRange().Start.getPointer();
    });

    for (auto &fixIt : sortedFixIts) {
      SMRange range = fixIt.getRange();
      unsigned offset = range.Start.getPointer() - bufferStart;
      unsigned length = range.End.getPointer() - range.Start.getPointer();

      doc.replacements.push_back(
          {filePath, offset, length, fixIt.getText().str()});
    }
  }

  // Write using LLVM's YAML output for proper escaping and formatting.
  llvm::Error err = llvm::writeToOutput(*exportPath, [&](raw_ostream &out) {
    llvm::yaml::Output yamlOut(out);
    yamlOut << doc;
    return llvm::Error::success();
  });
  if (err) {
    std::string errMsg;
    llvm::raw_string_ostream errOs(errMsg);
    errOs << err;
    return Error("Error writing YAML file '" + *exportPath + "': " + errMsg);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Diags implementation
//===----------------------------------------------------------------------===//

Diags::Diags(SourceMgr &sourceMgr, MLIRContext *context,
             bool useMLIRDiagnostics, int maxNotesPerDiagnostic,
             StringRef stripFilenamePrefix, bool disableWarnings,
             bool warningsAsErrors, void *extraContext,
             AutoFixItHandler *autoFixItHandler)
    : sourceMgr(sourceMgr), context(context), extraContext(extraContext),
      sourceMgrMapper(std::make_unique<SourceMgrLocationMapper>(
          context, stripFilenamePrefix)),
      useMLIRDiagnostics(useMLIRDiagnostics),
      autoFixItHandler(autoFixItHandler),
      maxNotesPerDiagnostic(maxNotesPerDiagnostic),
      disableWarnings(disableWarnings), warningsAsErrors(warningsAsErrors) {}

Diags::~Diags() {}

/// Return the identifier for the main buffer in the SourceMgr.
StringAttr Diags::getBufferNameIdentifier() const {
  if (!bufferNameIdentifier) {
    if (sourceMgr.getNumBuffers() == 0)
      return sourceMgrMapper->unknownBufferNameIdentifier;
    bufferNameIdentifier =
        sourceMgrMapper
            ->getFilenameFromBufferId(sourceMgr, sourceMgr.getMainFileID())
            .getAsOpaquePointer();
  }

  return StringAttr::getFromOpaquePointer(*bufferNameIdentifier);
}

/// Emit an error through the parser's logic.
InflightDiag Diags::emitError(Location loc, const Twine &message) {
  return InflightDiag(loc, *this, /*isWarning=*/false) << message;
}

/// Emit an error through the parser's logic.
InflightDiag Diags::emitError(llvm::SMLoc loc, const Twine &message) {
  return emitError(translateLocation(loc), message);
}

/// Emit a warning.
InflightDiag Diags::emitWarning(Location loc, const Twine &message) {
  return InflightDiag(loc, *this, /*isWarning=*/true) << message;
}
InflightDiag Diags::emitWarning(llvm::SMLoc loc, const Twine &message) {
  return emitWarning(translateLocation(loc), message);
}

/// Encode the specified source location information into a Location object
/// for attachment to the IR or error reporting.  This always returns a
/// FileLineColLoc.
FileLineColLoc Diags::translateLocation(SMLoc loc) const {
  return sourceMgrMapper->convertSMLocToLoc(sourceMgr, loc);
}

std::string Diags::getCanonicalFilename(StringRef filenameRef) const {
  return sourceMgrMapper->getCanonicalFilename(filenameRef);
}

SMLoc Diags::convertLocToSMLoc(LocationAttr loc) const {
  if (!loc)
    return SMLoc();
  if (FileLineColLoc fileLoc = loc.findInstanceOf<FileLineColLoc>())
    return sourceMgrMapper->convertLocToSMLoc(sourceMgr, fileLoc);
  return SMLoc();
}

void Diags::recordImportedFileIncludeLoc(LocationAttr loc, SMLoc includeLoc) {
  if (!loc || !includeLoc.isValid())
    return;
  // Derive the filename exactly as convertLocToSMLoc does, so we seed the same
  // filename that will later be looked up when the note's loc is converted.
  if (FileLineColLoc fileLoc = loc.findInstanceOf<FileLineColLoc>())
    sourceMgrMapper->recordIncludeLoc(fileLoc.getFilename(), includeLoc);
}

llvm::SMRange Diags::convertToSMRange(SourceRange range) const {
  SMRange byteLevelRange{range.getStart(), range.getEnd()};

  // SourceRange typically represents the end of range in terms of the start
  // of the end location.  Convert to a SMRange with a byte-level end position
  // if needed.
  if (!range.isByteLevel() && tokenEndPointAdjustmentFn &&
      byteLevelRange.End.isValid())
    tokenEndPointAdjustmentFn(byteLevelRange.End);
  return byteLevelRange;
}

void Diags::printImportStack(SMLoc includeLoc, raw_ostream &OS) const {
  if (includeLoc == SMLoc())
    return; // Top of stack.

  unsigned bufferID = sourceMgr.FindBufferContainingLoc(includeLoc);
  assert(bufferID && "Invalid or unspecified location!");
  printImportStack(sourceMgr.getBufferInfo(bufferID).IncludeLoc, OS);

  StringAttr bufferName =
      sourceMgrMapper->getFilenameFromBufferId(sourceMgr, bufferID);
  std::string canonicalBufferName = getCanonicalFilename(bufferName.str());
  auto lineAndColumn = sourceMgr.getLineAndColumn(includeLoc, bufferID);
  OS << "Imported from " << canonicalBufferName << ":" << lineAndColumn.first
     << ":\n";
}

void Diags::emitDiagnostic(const llvm::SMDiagnostic &diag) const {
  auto canonicalDiag = diag;

  StringRef origFilename = diag.getFilename();
  // Canonicalize the filename of the diag before printing. This strips any
  // file prefixes we've been asked to.
  if (!origFilename.empty() && origFilename != "-" &&
      origFilename != sourceMgrMapper->kUnnamedFileSigil) {
    std::string filename = getCanonicalFilename(origFilename);
    canonicalDiag = llvm::SMDiagnostic(
        *diag.getSourceMgr(), diag.getLoc(), filename, diag.getLineNo(),
        diag.getColumnNo(), diag.getKind(), diag.getMessage(),
        diag.getLineContents(), diag.getRanges(), diag.getFixIts());
  }

  if (auto diagHandler = sourceMgr.getDiagHandler()) {
    diagHandler(canonicalDiag, sourceMgr.getDiagContext());
    return;
  }

  if (canonicalDiag.getLoc().isValid()) {
    SMLoc loc = canonicalDiag.getLoc();
    unsigned bufferID = sourceMgr.FindBufferContainingLoc(loc);
    assert(bufferID && "Invalid or unspecified location!");
    printImportStack(sourceMgr.getBufferInfo(bufferID).IncludeLoc,
                     llvm::errs());
  } else if (!origFilename.empty()) {
    // The location has no buffer we could resolve - e.g., a precompiled
    // package's source file that isn't on disk. We may still know where that
    // package was imported into the user's program, so print that import stack
    // anyway.
    printImportStack(sourceMgrMapper->getRecordedIncludeLoc(origFilename),
                     llvm::errs());
  }

  canonicalDiag.print(nullptr, llvm::errs());
}

//===----------------------------------------------------------------------===//
// InflightDiag Implementation
//===----------------------------------------------------------------------===//

/// Each message in a diagnostic must have a location and text, and may
/// have any number of highlighted ranges, fixit hints, and line contents.
struct InflightDiag::Message {
  Location loc;
  std::string text;
  std::vector<SMRange> ranges;
  std::vector<SMFixIt> fixIts;
  // If non-empty, overrides the line contents automatically filled out by the
  // diagnostic manager.
  std::string customLineContents;
};

InflightDiag::InflightDiag(InflightDiag &&other)
    : messages(std::move(other.messages)), diags(other.diags),
      isWarning(other.isWarning) {
  // Do not emit the other diagnostic.
  other.diags = nullptr;
}

InflightDiag &InflightDiag::operator=(InflightDiag &&other) {
  messages = std::move(other.messages);
  diags = other.diags;
  isWarning = other.isWarning;

  // Do not emit the other diagnostic.
  other.diags = nullptr;
  return *this;
}

InflightDiag::InflightDiag(Location loc, Diags &diags, bool isWarning)
    : diags(&diags), isWarning(isWarning) {
  messages.push_back({loc, /*message=*/"", /*ranges=*/{}, /*fixIts=*/{},
                      /*customLineContents=*/{}});
}

InflightDiag::~InflightDiag() {
  // If the diagnostic got abandoned, just drop it.
  if (!diags)
    return;

  // Promote warning to error if -Werror is set.
  // This must happen BEFORE the disableWarnings check so that promoted
  // errors are not suppressed.
  if (diags->warningsAsErrors && isWarning)
    isWarning = false;

  // Note: At this point, if -Werror was set, isWarning is now false,
  // so this check won't suppress the promoted error.
  if (diags->disableWarnings && isWarning)
    return;

  diags->diagnosticEmitted = true;
  if (!isWarning)
    diags->errorEmitted = true;
  if (diags->useMLIRDiagnostics)
    emitMLIRDiagnostic();
  else if (diags->autoFixItHandler)
    emitAutoFixItDiagnostic();
  else
    emitSourceMgrDiagnostic();
}

/// Build the MLIR diagnostic and hand it off to its diagnostic machinery.
void InflightDiag::emitMLIRDiagnostic() {
  Location loc = messages.front().loc;
  InFlightDiagnostic mlirDiag =
      isWarning ? mlir::emitWarning(loc) : mlir::emitError(loc);
  mlirDiag << messages.front().text;
  // MLIR diagnostics can't print custom line text; just output it as yet
  // another note.
  if (!messages.front().customLineContents.empty())
    mlirDiag.attachNote(loc) << messages.front().customLineContents;
  for (auto &note : llvm::drop_begin(messages)) {
    mlirDiag.attachNote(note.loc) << note.text;
    if (!note.customLineContents.empty())
      mlirDiag.attachNote(note.loc) << note.customLineContents;
  }
}

void InflightDiag::emitAutoFixItDiagnostic() {
  auto &sourceMgr = diags->sourceMgr;

  // Register all fixits with the handler. We will resolve conflicts later.
  for (auto &message : messages) {
    for (SMFixIt &fixIt : message.fixIts) {
      unsigned bufferId =
          sourceMgr.FindBufferContainingLoc(fixIt.getRange().Start);
      const llvm::MemoryBuffer *buffer = sourceMgr.getMemoryBuffer(bufferId);
      diags->autoFixItHandler->registerFixIt(buffer, fixIt);
    }
  }

  // Following clang-tidy semantics: also print the diagnostic to the user.
  // The fix-it handler collects fixes for export/apply, but users should still
  // see the diagnostic messages.
  emitSourceMgrDiagnostic();
}

llvm::SMDiagnostic InflightDiag::getAsSMDiagnostic(
    const InflightDiag::Message &message, const std::string &text,
    SourceMgr::DiagKind kind, const std::string &customLineContents) {
  auto loc = diags->convertLocToSMLoc(message.loc);

  // GetMessage will automatically set the line contents of the diagnostic with
  // location information. We can't do this if the location isn't valid, or if
  // we have custom line contents we want to print.
  if (loc.isValid() && customLineContents.empty()) {
    // If we've got a valid location, get a fully fledged diagnostic with
    // location information, line text, etc.
    return diags->sourceMgr.GetMessage(loc, kind, text, message.ranges,
                                       message.fixIts);
  }

  // With an invalid location (e.g., the file is not available on disk), try
  // to construct something helpful for users.
  unsigned lineNo = -1, colNo = -1;
  SmallVector<std::pair<unsigned, unsigned>, 4> colRanges;
  StringRef bufferID = "<unknown>";

  if (loc.isValid()) {
    std::tie(lineNo, colNo) = diags->sourceMgr.getLineAndColumn(loc);
    unsigned curBuf = diags->sourceMgr.FindBufferContainingLoc(loc);
    assert(curBuf && "Invalid or unspecified location!");

    const llvm::MemoryBuffer *curMB = diags->sourceMgr.getMemoryBuffer(curBuf);
    bufferID = curMB->getBufferIdentifier();
  } else {
    // Walk the location structure and return the first FileLineColLoc we see.
    std::function<std::optional<FileLineColLoc>(Location)> visit =
        [&](Location l) -> std::optional<FileLineColLoc> {
      if (auto fileLoc = mlir::dyn_cast<FileLineColLoc>(l))
        return fileLoc;

      if (auto fused = dyn_cast<mlir::FusedLoc>(l)) {
        for (Location nested : fused.getLocations()) {
          if (auto fileLoc = visit(nested))
            return fileLoc;
        }
      }

      if (auto named = mlir::dyn_cast<mlir::NameLoc>(l))
        return visit(named.getChildLoc());

      if (auto call = mlir::dyn_cast<mlir::CallSiteLoc>(l)) {
        // Prefer the callee location
        if (auto fileLoc = visit(call.getCallee()))
          return fileLoc;
        return visit(call.getCaller());
      }

      return std::nullopt;
    };
    if (std::optional<FileLineColLoc> fileLoc = visit(message.loc)) {
      if (auto name = fileLoc->getFilename().getValue(); !name.empty()) {
        bufferID = name;
        lineNo = fileLoc->getLine();
        // Note: we may have a column number but here we leave it as -1.
        // Otherwise, if both line and column are not -1, the SMDiagnostic
        // prints a caret "into the file" which is likely unhelpful since it's
        // probably not on disk. With this tradeoff, we lose a valid column
        // number but get a normal-looking filename and line number without
        // the added noise of a caret into a path the user can't access
        // anyway:
        //   path/to/filename.mojo:100: note: foo
      }
    }
  }

  // If we've custom line text, we can't stop the diagnostic manager printing a
  // caret. Set the column for it back to 0 so it doesn't point somewhere odd.
  if (!customLineContents.empty())
    colNo = 0;

  return llvm::SMDiagnostic(diags->sourceMgr, loc, bufferID, lineNo, colNo,
                            kind, text, customLineContents, colRanges,
                            message.fixIts);
}

/// Print the diagnostic + each note through SourceMgr.
void InflightDiag::emitSourceMgrDiagnostic() {
  int nMessagesPrinted = 0;
  // The first diagnostic in the sequence is a warning or an error.
  SourceMgr::DiagKind kind =
      isWarning ? SourceMgr::DK_Warning : SourceMgr::DK_Error;
  for (auto &message : messages) {
    // Limit number of notes to print
    ++nMessagesPrinted;
    llvm::raw_string_ostream text(message.text);
    auto nOmitted = messages.size() - nMessagesPrinted;
    if (nMessagesPrinted > diags->maxNotesPerDiagnostic && nOmitted > 0)
      text << " (" << nOmitted << " more notes omitted.)";

    llvm::SMDiagnostic diag = getAsSMDiagnostic(message, text.str(), kind,
                                                message.customLineContents);

    diags->emitDiagnostic(diag);

    if (nMessagesPrinted > diags->maxNotesPerDiagnostic)
      break;

    // Subsequent diagnostics are all notes.
    kind = SourceMgr::DK_Note;
  }
}

/// Add a note to this diagnostic at the specified location, and change the
/// emission point to start filling it in.
InflightDiag InflightDiag::attachNote(Location loc) && {
  messages.push_back({loc, /*message=*/"", /*ranges=*/{}, /*fixIts=*/{},
                      /*customLineContents=*/{}});
  return std::move(*this);
}
InflightDiag &InflightDiag::attachNote(Location loc) & {
  messages.push_back({loc, /*message=*/"", /*ranges=*/{}, /*fixIts=*/{},
                      /*customLineContents=*/{}});
  return *this;
}

InflightDiag InflightDiag::attachNote(SMLoc loc) && {
  // If the diagnostic has been detached then we cannot translate the location,
  // but we don't care if we are anyway.
  if (!diags)
    return std::move(*this);
  return std::move(*this).attachNote(diags->translateLocation(loc));
}

InflightDiag &InflightDiag::attachNote(SMLoc loc) & {
  // If the diagnostic has been detached then we cannot translate the location,
  // but we don't care if we are anyway.
  if (!diags)
    return *this;
  return attachNote(diags->translateLocation(loc));
}

void InflightDiag::addDiag(InflightDiag &&otherDiag) {
  auto otherMessages = std::move(otherDiag.messages);
  Message &otherPrimary = otherMessages[0];
  Message &lastMsg = messages.back();
  lastMsg.text += otherPrimary.text;
  lastMsg.customLineContents += otherPrimary.customLineContents;
  llvm::append_range(lastMsg.ranges, std::move(otherPrimary.ranges));
  llvm::append_range(lastMsg.fixIts, std::move(otherPrimary.fixIts));
  llvm::append_range(messages, llvm::drop_begin(std::move(otherMessages)));
  otherDiag.abandon();
}

void InflightDiag::addText(const Twine &text) {
  messages.back().text += text.str();
}

void InflightDiag::addCustomLineText(const Twine &text) {
  messages.back().customLineContents += text.str();
}

static SMRange translateToSMRange(SourceRange range, Diags *diags) {
  if (!diags || diags->useMLIRDiagnostics)
    return {range.getStart(), range.getEnd()};
  return diags->convertToSMRange(range);
}

void InflightDiag::addSourceRange(SourceRange range) {
  messages.back().ranges.push_back(translateToSMRange(range, diags));
}

void InflightDiag::addFixIt(FixIt fixIt) {
  messages.back().fixIts.push_back(
      SMFixIt(translateToSMRange(fixIt.range, diags), fixIt.replacement));
}

/// Get the location of the most recent message in the diagnostic.
Location InflightDiag::getLastLoc() const { return messages.back().loc; }
Location InflightDiag::getPrimaryLoc() const { return messages.front().loc; }

//===----------------------------------------------------------------------===//
// FixIt Implementation
//===----------------------------------------------------------------------===//

FixIt::FixIt(SourceRange range, const Twine &replacement)
    : range(range), replacement(replacement.str()) {}

/// This constructor creates a fixit that removes the specified token.
FixIt FixIt::remove(SMLoc loc) { return FixIt({loc, loc}, Twine()); }

/// This constructor creates a fixit that removes the specified token range.
FixIt FixIt::remove(SourceRange range) { return FixIt(range, Twine()); }

/// This constructor creates a fixit that replaces the one token at the
/// specified location with some text.
FixIt FixIt::replaceToken(SMLoc loc, const Twine &text) {
  return FixIt({loc, loc}, text);
}

/// This constructor creates a fixit that inserts some text before the token
/// at the specified location, without replacing the token.
FixIt FixIt::insertBeforeToken(SMLoc loc, const Twine &text) {
  // Set the replacement range to an empty byte-level range before the token.
  return FixIt(SourceRange::getByteLevel(loc, loc), text);
}

/// This constructor creates a fixit that inserts some text after the token
/// at the specified location.
FixIt FixIt::insertAfterToken(SMLoc loc, const Twine &text, Diags &diags) {
  // Find end of token if we have a token end point adjustment function.
  if (diags.tokenEndPointAdjustmentFn && loc.isValid())
    diags.tokenEndPointAdjustmentFn(loc);
  return FixIt(SourceRange::getByteLevel(loc, loc), text);
}

//===----------------------------------------------------------------------===//
// addToDiagnostic helpers
//===----------------------------------------------------------------------===//

// Allow inserting string-like things.
void M::addToDiagnostic(const Twine &text, InflightDiag &diag) {
  diag.addText(text);
}

void M::addToDiagnostic(char text, InflightDiag &diag) {
  diag.addText(Twine(text));
}

void M::addToDiagnostic(size_t number, InflightDiag &diag) {
  diag.addText(Twine(number));
}

void M::addToDiagnostic(StringAttr attr, InflightDiag &diag) {
  diag.addText(Twine("'"));
  diag.addText(attr.getValue());
  diag.addText(Twine("'"));
}

/// This adds a source range highlight.
void M::addToDiagnostic(SourceRange range, InflightDiag &diag) {
  diag.addSourceRange(range);
}

/// This adds a fixit hint.
void M::addToDiagnostic(FixIt fixIt, InflightDiag &diag) {
  diag.addFixIt(fixIt);
}

void M::addToDiagnostic(InflightDiag &&otherDiag, InflightDiag &diag) {
  diag.addDiag(std::move(otherDiag));
}

M::ScopedMLIRWarningHandler::ScopedMLIRWarningHandler(MLIRContext *ctx,
                                                      bool disableWarnings,
                                                      bool warningsAsErrors)
    : handler(ctx,
              [=, errorEmittedPtr = &errorEmitted](mlir::Diagnostic &diag) {
                if (diag.getSeverity() == mlir::DiagnosticSeverity::Warning) {
                  // Promote warning to error if -Werror is set
                  if (warningsAsErrors) {
                    mlir::emitError(diag.getLocation()) << diag.str();
                    *errorEmittedPtr = true;
                    return mlir::success(); // Suppress original warning
                  }

                  // Suppress warnings if --disable-warnings is set
                  if (disableWarnings)
                    return mlir::success();
                }
                return mlir::failure(); // Pass through
              }) {}
