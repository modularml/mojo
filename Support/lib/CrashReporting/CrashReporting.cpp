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

#include "Support/CrashReporting/CrashReporting.h"

#include "Config/Version.h"
#include "Support/Configuration.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <utility>

#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "client/crash_report_database.h"
#include "client/crashpad_client.h"
#include "client/settings.h"
#include "client/simulate_crash.h" // IWYU pragma: keep (CRASHPAD_SIMULATE_CRASH)

using namespace M;

static constexpr llvm::StringLiteral kHandlerProgramName =
    "modular-crashpad-handler";
static constexpr llvm::StringLiteral kDefaultURL =
    "https://crash-reporting.modular.com";

std::filesystem::path
M::getCrashDatabasePath(const std::filesystem::path &dataFolder) {
  return dataFolder / "crashdb";
}

ErrorOr<std::filesystem::path> M::getCrashpadHandlerPath(Config *settings) {
  StringRef program("");
  if (settings) {
    program = settings->getValue("crash_reporting.handler_path");
  }
  std::string foundProgram;
  if (program.empty()) {
    // N.B.: Errors from findProgramByName are intentionally ignored for the
    // same reason as above.
    if (auto programOr = llvm::sys::findProgramByName(kHandlerProgramName)) {
      foundProgram = std::move(*programOr);
      program = foundProgram;
    }
  }
  // No luck.
  if (program.empty())
    return Error("unable to locate crashpad handler executable");
  return std::string_view(program);
}

static ErrorOrSuccess tryInitCrashpad(StringRef program, StringRef machineID,
                                      StringRef sessionID, Config *settings) {
  // Crashpad needs a few paths and other configuration bits:
  //   - Path of the handler executable (This runs alongside the Mojo driver;
  //     in case the driver crashes, the handler inspects the driver in its
  //     crashed state and generates a crash report)
  //   - Crash database, to put the crashes in before they are sent off
  //   - URL to upload crash reports to
  auto dataFolderOr = Config::getModularDataFolderPath();
  if (dataFolderOr.isError())
    return dataFolderOr.takeError();
  std::filesystem::path databasePath = getCrashDatabasePath(*dataFolderOr);

  auto handlerPathOr = getCrashpadHandlerPath(settings);
  if (handlerPathOr)
    return Error(llvm::Twine("while locating crashpad handler: ") +
                 handlerPathOr.getError());
  std::filesystem::path handlerPath = std::move(*handlerPathOr);
  std::string url("");
  if (settings)
    url = settings->getValue("crash_reporting.url");
  if (url.empty())
    url = kDefaultURL.str();
  assert(!url.empty() && "Crashpad URL must not be empty.");
  url = llvm::Twine(url + "/" + program).str();

  // Update the database if reporting is not enabled. In most
  // cases this will just read the existing database settings and not change.
  auto database =
      crashpad::CrashReportDatabase::Initialize(base::FilePath(databasePath));
  bool uploadsEnabled = false;
  if (database != nullptr && database->GetSettings() != nullptr &&
      (!database->GetSettings()->GetUploadsEnabled(&uploadsEnabled) ||
       !uploadsEnabled))
    database->GetSettings()->SetUploadsEnabled(true);

  // Setup all the annotations.
  std::map<std::string, std::string> annotations;
  annotations["program"] = std::string(program);
  annotations["version"] = getModularVersionString();
  // Must match the usage telemetry lane's machineid/sessionid resource
  // attributes so crash reports can be joined with usage events.
  annotations["machineid"] = std::string(machineID);
  annotations["sessionid"] = std::string(sessionID);

  // Launch Crashpad handler.
  crashpad::CrashpadClient client;
  if (!client.StartHandler(
          base::FilePath(handlerPath), base::FilePath(databasePath),
          /*metrics_dir=*/base::FilePath(databasePath), std::string(url),
          /*annotations=*/annotations,
          /*arguments=*/{"--no-rate-limit"}, /*restartable=*/true,
          /*asynchronous_start=*/false))
    return Error("crashpad failed to start handler");
  return success();
}

void M::initCrashpadForProgram(StringRef program, StringRef machineID,
                               StringRef sessionID, Config *settings) {
  if (auto error = tryInitCrashpad(program, machineID, sessionID, settings))
    llvm::errs() << "Failed to initialize Crashpad.  "
                    "Crash reporting will not be available.  "
                    "Cause: "
                 << error.getError() << "\n";
}

void M::generateNonFatalDump() { CRASHPAD_SIMULATE_CRASH(); }
