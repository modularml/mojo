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

#include "Support/Compiler/ErrorTree.h"
#include "Support/Compiler/Error.h"
#include "Support/Error.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/STLExtras.h"
#include <cassert>
#include <optional>
#include <utility>
#include <vector>

using namespace M;

ErrorTree::ErrorTree(Location loc, Error error, ErrorTree causes)
    : loc(loc), error(std::move(error)) {
  addCause(std::move(causes));
}

ErrorTree::ErrorTree(Location loc, Error error,
                     MutableArrayRef<ErrorTree> causes)
    : loc(loc), error(std::move(error)) {
  addCauses(causes);
}

ErrorTree ErrorTree::copy() const {
  ErrorTree copy(loc, error.copy());
  copy.causes.reserve(causes.size());
  for (const ErrorTree &cause : causes)
    copy.causes.push_back(cause.copy());
  return copy;
}

/// This function attempts detect recursive errors along a single-element branch
/// of the error tree and collapse them to make the errors more digestible.
/// Specifically, the function looks at single-element segments of the error
/// tree and runs repeating subsequence detection.
///
/// Given a finite sequence, find the largest non-overlapping subsequences that
/// are repeats of smaller subsequences.
static void bundleRecursiveErrors(
    std::vector<ErrorTree *> &path,
    DenseMap<std::pair<Location, StringRef>, ErrorTree *> &seen, ErrorTree *cur,
    ErrorTree *parent, int last, int prog, int start) {
  // Run cycle detection.
  if (last == -1 &&
      !seen.try_emplace({cur->getLoc(), cur->getMessage()}, parent).second) {
    // Backtrack the path to find the last matching error in the path.
    for (int i = path.size() - 1; i >= 0; --i) {
      ErrorTree *prev = path[i];
      if (prev->getLoc() != cur->getLoc() ||
          prev->getMessage() != cur->getMessage())
        continue;
      last = i;
      prog = i;
      break;
    }
    assert(last != -1 && "no matching previous?");
    start = path.size();
  }

  // Iterate to the next element in the sequence.
  path.push_back(cur);

  if (last != -1) {
    // Attempt to progress.
    ErrorTree *tentative = path[prog];
    if (tentative->getLoc() != cur->getLoc() ||
        tentative->getMessage() != cur->getMessage()) {
      // TODO: The algorithm could continue to backtrack here, allowing it to
      // detect repeated subsequences like `*, A, B, A, D, A, B, A, D, *`.

      // Compute the period and the offset from the start of the cycle.
      int period = start - last;
      int curDiff = ((prog - last) + start) - last;
      // Round down to the nearest multiple of the period.
      curDiff = curDiff / period * period;

      // We have the subsequence that is comprised of a repeated subsequence.
      // Bundle up the errors in that sequence.
      int reps = curDiff / period;
      ErrorTree *base = path[last];
      ErrorTree *tip = path[last + curDiff];
      auto it = seen.find({base->getLoc(), base->getMessage()});
      ErrorTree *parent = it->second;
      ErrorTree bundle(
          base->getLoc(),
          "error recurses " + Twine(reps) + " times:", std::move(*base));
      ErrorTree rest(base->getLoc(),
                     "remaining errors after:", std::move(*tip));
      path[last + period - 1]->getCauses().clear();
      parent->getCauses().clear();
      parent->addCause(std::move(bundle));
      parent->addCause(std::move(rest));

      // Reset cycle detection state and restart at the tip.
      seen.clear();
      path.clear();
      cur = &parent->getCauses().back();
      last = -1;
      prog = -1;
      start = -1;
    } else {
      ++prog;
    }
  }

  if (cur->getCauses().size() == 1) {
    // Continue iterating.
    bundleRecursiveErrors(path, seen, &cur->getCauses().front(), cur, last,
                          prog, start);
  } else {
    // Reset recursion and visit the new single-element segments.
    for (ErrorTree &err : cur->getCauses()) {
      std::vector<ErrorTree *> newPath;
      DenseMap<std::pair<Location, StringRef>, ErrorTree *> newSeen;
      bundleRecursiveErrors(newPath, newSeen, &err, cur, -1, -1, -1);
    }
  }
}

/// Dig out a CallSiteLoc from the given location.
static std::optional<mlir::CallSiteLoc> getCallSiteLoc(Location loc) {
  if (auto name = dyn_cast<mlir::NameLoc>(loc))
    return getCallSiteLoc(name.getChildLoc());
  if (auto callLoc = dyn_cast<mlir::CallSiteLoc>(loc))
    return callLoc;
  if (auto fused = dyn_cast<FusedLoc>(loc)) {
    for (auto subLoc : fused.getLocations()) {
      if (auto callLoc = getCallSiteLoc(subLoc))
        return callLoc;
    }
    return {};
  }
  return {};
}

static void emitErrorTreeDiag(const ErrorTree &err,
                              function_ref<void(Location, StringRef)> emit,
                              StringRef callSiteMsg) {
  Location loc = err.getLoc();

  SmallVector<Location> locationStack{loc};
  for (std::optional<mlir::CallSiteLoc> callLoc;
       (callLoc = getCallSiteLoc(loc)); loc = callLoc->getCallee())
    locationStack.push_back(callLoc->getCaller());
  if (locationStack.empty()) {
    emit(loc, err.getMessage());
  } else {
    for (Location loc : llvm::drop_begin(locationStack))
      emit(loc, callSiteMsg);
    emit(locationStack.front(), err.getMessage());
  }
}

InFlightDiagnostic ErrorTree::emit(
    function_ref<InFlightDiagnostic(Location)> emitError, StringRef callSiteMsg,
    bool emitPrelude,
    std::optional<mlir::DiagnosticEngine::HandlerID> diagHandlerID) && {
  // Try to compress recursive errors. To provide a root, start iterating from
  // the first child.
  for (ErrorTree &cause : causes) {
    std::vector<ErrorTree *> path;
    DenseMap<std::pair<Location, StringRef>, ErrorTree *> seen;
    bundleRecursiveErrors(path, seen, &cause, this, -1, -1, -1);
  }

  // Emit the main error.
  std::optional<InFlightDiagnostic> diag;
  auto emitMsg = [&](Location loc, StringRef msg) {
    if (!emitPrelude && isLocationInPrelude(loc))
      return;
    if (diag)
      diag->attachNote(loc) << msg;
    else
      diag.emplace(emitError(loc)) << msg;
  };
  emitErrorTreeDiag(*this, emitMsg, callSiteMsg);

  // Add a DiagnosticsArgument to indicate that this one is from an
  // InFlightDiagnostic so that it can be filtered later by DiagnosticHandler.
  if (diagHandlerID) {
    if (mlir::Diagnostic *underlyingDiag = diag->getUnderlyingDiagnostic()) {
      underlyingDiag->getMetadata().push_back(
          mlir::DiagnosticArgument(*diagHandlerID));
    }
  }

  // Emit the causes.
  emit(diag, causes, callSiteMsg, emitPrelude);
  if (!diag.has_value()) {
    diag.emplace(emitError(mlir::UnknownLoc()))
        << "error happened but nothing is emitted with the diagnostics, please "
           "use -elaboration-error-include-prelude to include prelude errors";
  }

  return std::move(*diag);
}

void ErrorTree::emit(std::optional<InFlightDiagnostic> &diag,
                     ArrayRef<ErrorTree> errors, StringRef callSiteMsg,
                     bool emitPrelude) {
  if (errors.empty())
    return;

  for (const ErrorTree &err : errors) {
    emitErrorTreeDiag(
        err,
        [&](Location loc, StringRef msg) {
          if (!emitPrelude && isLocationInPrelude(loc))
            return;
          if (diag)
            diag->attachNote(loc) << msg;
          else
            diag.emplace(emitError(loc)) << msg;
        },
        callSiteMsg);
    emit(diag, err.causes, callSiteMsg, emitPrelude);
  }
}
