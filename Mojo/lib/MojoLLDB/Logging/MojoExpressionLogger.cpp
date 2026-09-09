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

#include "MojoExpressionLogger.h"
#include "../ExpressionParser/MojoDiagnostic.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/LLDBLog.h"
#include "llvm/ADT/SmallVector.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::Mojo;
using namespace lldb_private;

MojoExpressionLogger::MojoExpressionLogger(Target &target)
    : Broadcaster(target.GetDebugger().GetBroadcasterManager(),
                  "mojo-expression.broadcaster") {}

void MojoExpressionLogger::broadcastUserMessage(StringRef message) {
  lldb::EventSP event = std::make_shared<Event>(eBroadcastUserMessage,
                                                new EventDataBytes(message));
  BroadcastEvent(event);
}

void MojoExpressionLogger::dumpIR(StringRef message) {
  lldb::EventSP event =
      std::make_shared<Event>(eDumpIR, new EventDataBytes(message));
  BroadcastEvent(event);
}

void MojoExpressionLogger::debugLog(StringRef message) {
  lldb::EventSP event =
      std::make_shared<Event>(eDebugLog, new EventDataBytes(message));
  BroadcastEvent(event);
}

void MojoExpressionLogger::errorLog(StringRef message) {
  lldb::EventSP event =
      std::make_shared<Event>(eErrorLog, new EventDataBytes(message));
  BroadcastEvent(event);
}

void MojoExpressionLogger::broadcastDiagnostics(
    DiagnosticManager &diagnosticManager,
    function_ref<bool(MojoDiagnostic &)> filter) {
  debugLog("Emitted diagnostics");

  std::string msg;
  llvm::raw_string_ostream msgOS(msg);
  for (const auto &diag : diagnosticManager.Diagnostics()) {
    if (auto *mojoDiag = dyn_cast<MojoDiagnostic>(diag.get())) {
      if (filter && !filter(*mojoDiag))
        continue;
    }

    switch (diag->GetSeverity()) {
    case lldb::eSeverityError:
      // Log error diagnostics explicitly so they get captured in the error log.
      errorLog(("error: " + diag->GetMessage()).str());
      continue;
    case lldb::eSeverityWarning:
      msgOS << "warning: ";
      break;
    case lldb::eSeverityInfo:
      break;
    }
    msgOS << diag->GetMessage() << "\n";
  }
  if (!msg.empty())
    broadcastUserMessage(msg);
}

/// Get a null-terminated string from an event.
static std::string getStringFromEvent(const lldb::EventSP &event) {
  size_t readLen = EventDataBytes::GetByteSizeFromEvent(event.get());
  const char *rawData =
      static_cast<const char *>(EventDataBytes::GetBytesFromEvent(event.get()));
  return {rawData, readLen};
}

/// Stringify the event type.
static std::string stringifyType(MojoExpressionLogger::MessageKind type) {
  SmallVector<std::string, 1> typeStrs;
  if (type & MojoExpressionLogger::eBroadcastUserMessage)
    typeStrs.push_back("BroadcastUser");
  if (type & MojoExpressionLogger::eDumpIR)
    typeStrs.push_back("DumpIR");
  if (type & MojoExpressionLogger::eDebugLog)
    typeStrs.push_back("DebugLog");
  if (type & MojoExpressionLogger::eErrorLog)
    typeStrs.push_back("ErrorLog");

  std::string out;
  llvm::raw_string_ostream outStream(out);
  llvm::interleave(typeStrs, outStream, "|");
  return out;
}

void MojoExpressionLogger::handleEvent(
    const lldb::EventSP &event,
    function_ref<void(StringRef, StringRef)> sendUserOutput) {
  assert(llvm::popcount(event->GetType()) == 1 &&
         "a message must contain one single type");

  if (event->GetType() & MojoExpressionLogger::eBroadcastUserMessage) {
    // If it's a user message broadcast, send that output.
    sendUserOutput("user", getStringFromEvent(event));
  } else if (event->GetType() & (MojoExpressionLogger::eErrorLog)) {
    // If it's an error log, send that output as well.
    sendUserOutput("error", getStringFromEvent(event));
    LLDB_LOG(GetLog(LLDBLog::Expressions), "[{0}] {1}",
             stringifyType(MessageKind(event->GetType())),
             getStringFromEvent(event));
  } else if (event->GetType() & (eDumpIR | eDebugLog)) {
    Log *log = GetLog(LLDBLog::Expressions);
    if (!log)
      return;
    // DumpIR messages are extremely heavy, so we don't want to log them unless
    // verbose logs are enabled.
    if ((event->GetType() & eDumpIR) && !log->GetVerbose())
      return;
    LLDB_LOG(log, "[{0}] {1}", stringifyType(MessageKind(event->GetType())),
             getStringFromEvent(event));
  } else {
    llvm_unreachable("Unexpected message type");
  }
}

MojoExpressionLogger &
MojoExpressionLogger::getLoggerForTarget(lldb_private::Target &target) {
  // It's fine to keep this map around for the entire duration of the debug
  // session because the number of targets is small (almost always 2).
  static DenseMap<Target *, std::unique_ptr<MojoExpressionLogger>> loggerMap;
  auto it = loggerMap.find(&target);
  if (it == loggerMap.end())
    it = loggerMap
             .insert({&target, std::make_unique<MojoExpressionLogger>(target)})
             .first;
  return *it->getSecond();
}
