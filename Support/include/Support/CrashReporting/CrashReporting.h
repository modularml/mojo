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
// Facade in front of Crashpad for crash reporting.  This is not a part of the
// MSupport CMake target, instead you need to explicitly link against
// MCrashReporting.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_CRASHREPORTING_H
#define SUPPORT_CRASHREPORTING_H

#include "Support/Configuration.h"
#include "Support/ForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include <filesystem>

namespace M {

class Config;

/// Attempt to locate the Crashpad handler executable.
///
/// If specified in the configuration, that takes precedence.  Otherwise, we
/// look alongside the running executable, or failing that, anywhere on the
/// PATH.
ErrorOr<std::filesystem::path> getCrashpadHandlerPath(Config *settings);

/// Pick a location to store crash data in.
///
/// Returns the "crashdb" directory inside of the modular passed dataPath
std::filesystem::path
getCrashDatabasePath(const std::filesystem::path &dataPath);

/// Initialize crash reporting for currently running executable.
///
/// Note that this makes fairly invasive changes to the process environment
/// (removing existing signal handlers and adding new ones, spawning a
/// subprocess (potentially interfering with SIGCHLD handling), modifying the
/// process-global exception port on Darwin, etc) so it should only be called
/// from code that reasonably "owns" the process, not from a library where we
/// don't know what the rest of the code in the process is doing.
///
/// The argv0 parameter should be the first available argument (e.g. argv[0])
/// and may be used to help locate the crashpad binary (if there is no relevant
/// configuration). The program parameter is used for metadata when posting to
/// the crashpad API, which allows for clustering crashes server-side for
/// analysis; this should be simple and fixed (e.g. "mojo" is a good name).
/// The machineID and sessionID parameters are attached so crash reports
/// can be joined with usage events.
void initCrashpadForProgram(StringRef program, StringRef machineID,
                            StringRef sessionID, Config *settings = nullptr);

/// Generate a crash dump with the current state of the process, without
/// actually causing the current process to crash and terminate.
/// initCrashpadForProgram must have been previously called.
void generateNonFatalDump();

} // namespace M

#endif // SUPPORT_CRASHREPORTING_H
