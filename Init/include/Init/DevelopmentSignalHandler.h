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

#ifndef INIT_DEVELOPMENT_SIGNAL_HANDLER_H
#define INIT_DEVELOPMENT_SIGNAL_HANDLER_H

namespace llvm {
class StringRef;
}

namespace M::Init {

/// Register development signal handlers for crash reporting and debugging.
/// Only active in non-production builds. This function sets up comprehensive
/// signal handling for crash-like signals including SIGSEGV, SIGABRT, SIGFPE,
/// SIGILL, SIGBUS, SIGTRAP, and SIGSYS.
///
/// The signal handlers capture detailed signal information including signal
/// codes, fault addresses, and process information, then chain to LLVM's
/// signal handling infrastructure for stack traces and final cleanup.
void registerDevelopmentSignalHandler(llvm::StringRef programName);

/// Enable Python stack traces in signal handlers using async-safe faulthandler.
/// This configures the signal handler to use SIGUSR2 to trigger Python stack
/// traces without GIL deadlock risks.
void enablePythonStackTraceCallback();

} // namespace M::Init

#endif // INIT_DEVELOPMENT_SIGNAL_HANDLER_H
