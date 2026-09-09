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

#include "Support/Driver/DiagnosticFormat.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;
using namespace M::json;

llvm::json::Value M::json::toJSON(DiagnosticKind kind) {
  switch (kind) {
  case DiagnosticKind::Note:
    return "note";
  case DiagnosticKind::Warning:
    return "warning";
  case DiagnosticKind::Error:
    return "error";
  }
}

llvm::json::Value M::json::toJSON(const DiagnosticRange &range) {
  return llvm::json::Object{{"start", range.start}, {"end", range.end}};
}

llvm::json::Value M::json::toJSON(const DiagnosticLocation &location) {
  return llvm::json::Object{{"line", location.line},
                            {"column", location.column}};
}

llvm::json::Value M::json::toJSON(const FixIt &fixIt) {
  return llvm::json::Object{{"text", fixIt.text},
                            {"start", toJSON(fixIt.start)},
                            {"end", toJSON(fixIt.end)}};
}

llvm::json::Value M::json::toJSON(const SourceDiagnostic &diagnostic) {
  llvm::SmallVector<llvm::json::Value> ranges;
  ranges.reserve(diagnostic.ranges.size());
  for (const auto &range : diagnostic.ranges)
    ranges.push_back(toJSON(range));

  llvm::SmallVector<llvm::json::Value> fixIts;
  fixIts.reserve(diagnostic.fixIts.size());
  for (const auto &fixIt : diagnostic.fixIts)
    fixIts.push_back(toJSON(fixIt));

  llvm::json::Object result{{"file", diagnostic.file},
                            {"text", diagnostic.text},
                            {"location", toJSON(diagnostic.location)},
                            {"ranges", llvm::json::Array(ranges)},
                            {"fixIts", llvm::json::Array(fixIts)}};
  return result;
}

llvm::json::Value M::json::toJSON(const Diagnostic &diagnostic) {
  llvm::json::Object result{{"kind", toJSON(diagnostic.kind)},
                            {"message", diagnostic.message}};
  if (diagnostic.diagnostic)
    result["diagnostic"] = toJSON(*diagnostic.diagnostic);
  return result;
}

static DiagnosticKind toDiagnosticKind(llvm::SourceMgr::DiagKind kind) {
  switch (kind) {
  case llvm::SourceMgr::DK_Remark:
  case llvm::SourceMgr::DK_Note:
    return DiagnosticKind::Note;
  case llvm::SourceMgr::DK_Warning:
    return DiagnosticKind::Warning;
  case llvm::SourceMgr::DK_Error:
    return DiagnosticKind::Error;
  }
}

static Diagnostic toJSONDiagnostic(const llvm::SMDiagnostic &diag) {
  Diagnostic jsonDiag{toDiagnosticKind(diag.getKind()),
                      diag.getMessage().str()};
  // If the diagnostic doesn't include any valid location information, then
  // we're done.
  if (diag.getLineNo() == -1 || diag.getColumnNo() == -1)
    return jsonDiag;

  // Otherwise, collect information about the file and source location of the
  // diagnostic.
  SourceDiagnostic locDiag{
      diag.getFilename().str(),
      diag.getLineContents().str(),
      DiagnosticLocation{diag.getLineNo(), diag.getColumnNo()},
      {},
      {},
  };
  locDiag.ranges.reserve(diag.getRanges().size());
  for (const auto &range : diag.getRanges())
    locDiag.ranges.push_back(DiagnosticRange{range.first, range.second});

  // Fix-its only have source location pointers to the start and end of the text
  // replace. Translate these into offsets into the source line included in the
  // diagnostic.
  const llvm::SourceMgr *sourceMgr = diag.getSourceMgr();
  locDiag.fixIts.reserve(diag.getFixIts().size());
  for (const auto &fixIt : diag.getFixIts()) {
    auto start = sourceMgr->getLineAndColumn(fixIt.getRange().Start);
    auto end = sourceMgr->getLineAndColumn(fixIt.getRange().End);

    locDiag.fixIts.push_back(FixIt{
        fixIt.getText().str(), DiagnosticLocation{start.first, start.second},
        DiagnosticLocation{end.first, end.second}});
  }

  jsonDiag.diagnostic = locDiag;
  return jsonDiag;
}

/// An `llvm::SourceMgr` diagnostic handler that prints diagnostics as JSON to
/// stderr. Each diagnostic appears on its own line.
static void JSONDiagHandler(const llvm::SMDiagnostic &diagnostic,
                            void *context) {
  llvm::errs() << toJSON(toJSONDiagnostic(diagnostic)) << '\n';
}

llvm::SourceMgr::DiagHandlerTy M::getDiagHandler(DiagnosticFormat format) {
  switch (format) {
  case DiagnosticFormat::Text:
    // A null handler is the `llvm::SourceMgr` default. Setting a null handler
    // on a source manager has it print diagnostics as normal.
    return nullptr;
  case DiagnosticFormat::JSON:
    return JSONDiagHandler;
  }
}
