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

#include "Init/Init.h"
#include "AsyncRT/Runtime/HostSystem.h"
#include "Init/DevelopmentSignalHandler.h"
#include "Support/Configuration.h"
#include "Support/ContextGlobal.h"
#include "Support/CrashReporting/CrashReporting.h"
#include "Support/Telemetry/Telemetry.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Process.h"

#include <mutex>

using namespace M;

static constexpr bool isProductionBuild() {
#ifdef MODULAR_PRODUCTION
  return true;
#else
  return false;
#endif
}

namespace {

/// Internal function for creating an M::Context, not thread-safe, and not
/// intended for use outside of this file.
ErrorOr<ContextRef> createContextImpl(StringRef programName,
                                      const Init::Options &options,
                                      StringRef subCommand) {
  // Create the top-level context.
  ContextRef ctx = ContextRef::create();

  // Create the settings object.
  auto settingsOr = Config::open();
  if (settingsOr.isError())
    return settingsOr.takeError();
  Config settings = std::move(*settingsOr);

  bool crashReportingEnabled = Telemetry::isCrashReportingEnabled(settings);

  // Enable crash logging, if appropriate.
  if (!isProductionBuild() && !crashReportingEnabled)
    Init::registerDevelopmentSignalHandler(programName);
  else if (!options.forceDisableCrashReportingEnabled() &&
           crashReportingEnabled) {
    // The crash lane must carry the same machine/session IDs as the usage
    // telemetry lane so crash reports can be joined with usage events.
    const auto &localIDs = Telemetry::createLocalIDs();
    initCrashpadForProgram(programName, localIDs.machine, localIDs.session,
                           &settings);
  }

  // TelemetryContext created and added to the Context here as Config is
  // required.
  ctx->emplace<Telemetry::TelemetryContext>(settings, programName, subCommand);

  // Create a new cpuDevice (if needed).
  if (options.getCPUDeviceOptions()) {
    std::string profileFilename =
        llvm::sys::Process::GetEnv("MODULAR_PROFILE_FILENAME").value_or("");
    AsyncRT::CPUDeviceOptions opts = *options.getCPUDeviceOptions();
    if (!profileFilename.empty())
      opts.profileFilename = profileFilename;
    AsyncRT::CPUDeviceRef ref = AsyncRT::getOrCreateCPUDevice(
        AsyncRT::CPUDeviceSource::MaxContext, opts);
    ctx->setRuntime(GenericRCRef::fromRCRef(std::move(ref)));
  }

  // Finally move the settings.
  ctx->emplace<Config>(std::move(settings));

  // Store a copy of the init options so we can compare when reusing the
  // global context.
  ctx->emplace<Init::Options>(options);

  // Return the useable context.
  return std::move(ctx);
}

} // namespace

ErrorOr<ContextRef> Init::createContext(StringRef programName,
                                        const Init::Options &options,
                                        StringRef subCommand) {
  std::lock_guard<std::mutex> lock(getGlobalContextMutex());
  if (getCurrentMaxContextPointerOrNull()) {
    llvm::report_fatal_error(
        "Init::createContext() attempted to create a new M::Context when one"
        "already exists, considering using Init::getContext() instead.");
  }

  // Create context.
  auto ctxOr = createContextImpl(programName, options, subCommand);
  if (ctxOr.isError())
    return ctxOr.takeError();

  // Set as the global current context.
  setCurrentMaxContextPointer(ctxOr->getPointer());
  return ctxOr;
}

ErrorOr<ContextRef> Init::getOrCreateContext(StringRef programName,
                                             const Init::Options &options,
                                             StringRef subCommand) {
  std::lock_guard<std::mutex> lock(getGlobalContextMutex());
  if (Context *existing = getCurrentMaxContextPointerOrNull()) {
    if (*existing->get<Init::Options>() != options)
      llvm::report_fatal_error(
          "Init::getOrCreateContext() requested an M::Context with different"
          "Init::Options to those used to create the existing M::Context, check"
          "the options that are being requested.");
    return ContextRef::copy(existing);
  }

  // Create context.
  auto ctxOr = createContextImpl(programName, options, subCommand);
  if (ctxOr.isError())
    return ctxOr.takeError();

  // Set as the global current context.
  setCurrentMaxContextPointer(ctxOr->getPointer());
  return ctxOr;
}

ContextRef Init::getContext() {
  std::lock_guard<std::mutex> lock(getGlobalContextMutex());
  Context *existing = getCurrentMaxContextPointerOrNull();
  if (!existing) {
    llvm::report_fatal_error(
        "Init::getContext() requested the M::Context but no context has been"
        "created yet, Init::createContext() must be called to create the"
        "M::Context before calling Init::getContext().");
  }

  return ContextRef::copy(existing);
}
