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

#include "Init/DevelopmentSignalHandler.h"
#include "Support/MArchTarget/Host.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/Threading.h"

#include <csignal>
#include <ctime>
#include <sstream>
#include <sys/ucontext.h>
#include <thread>
#include <unistd.h>

using namespace M;

namespace {

// Flag to indicate if Python stack trace handling is available
// This is set when faulthandler is properly registered for SIGUSR2
static bool pythonStackTraceEnabled = false;

struct SignalInfo {
  std::string info;
  int signal = 0;
};
} // namespace

static SignalInfo &signalInformation() {
  // Avoid the need for a global static destructor.
  static SignalInfo signalInfo;
  return signalInfo;
}

static const char *getSignalName(int sig) {
  switch (sig) {
  case SIGSEGV:
    return "SIGSEGV";
  case SIGABRT:
    return "SIGABRT";
  case SIGFPE:
    return "SIGFPE";
  case SIGILL:
    return "SIGILL";
  case SIGBUS:
    return "SIGBUS";
  case SIGTRAP:
    return "SIGTRAP";
  case SIGSYS:
    return "SIGSYS";
  default:
    return "UNKNOWN";
  }
}

static const char *getSignalDescription(int sig) {
  switch (sig) {
  case SIGSEGV:
    return "Segmentation fault";
  case SIGABRT:
    return "Abort signal";
  case SIGFPE:
    return "Floating point exception";
  case SIGILL:
    return "Illegal instruction";
  case SIGBUS:
    return "Bus error";
  case SIGTRAP:
    return "Trace/breakpoint trap";
  case SIGSYS:
    return "Bad system call";
  default:
    return "Unknown signal";
  }
}

static const char *getSignalCodeDescription(int sig, int code) {
  switch (sig) {
  case SIGSEGV:
    switch (code) {
    case SEGV_MAPERR:
      return "Address not mapped to object";
    case SEGV_ACCERR:
      return "Invalid permissions for mapped object";
    default:
      return "Unknown segmentation fault cause";
    }
  case SIGFPE:
    switch (code) {
    case FPE_INTDIV:
      return "Integer divide by zero";
    case FPE_INTOVF:
      return "Integer overflow";
    case FPE_FLTDIV:
      return "Floating point divide by zero";
    case FPE_FLTOVF:
      return "Floating point overflow";
    case FPE_FLTUND:
      return "Floating point underflow";
    case FPE_FLTRES:
      return "Floating point inexact result";
    case FPE_FLTINV:
      return "Floating point invalid operation";
    case FPE_FLTSUB:
      return "Subscript out of range";
    default:
      return "Unknown floating point exception";
    }
  case SIGILL:
    switch (code) {
    case ILL_ILLOPC:
      return "Illegal opcode";
    case ILL_ILLOPN:
      return "Illegal operand";
    case ILL_ILLADR:
      return "Illegal addressing mode";
    case ILL_ILLTRP:
      return "Illegal trap";
    case ILL_PRVOPC:
      return "Privileged opcode";
    case ILL_PRVREG:
      return "Privileged register";
    case ILL_COPROC:
      return "Coprocessor error";
    case ILL_BADSTK:
      return "Internal stack error";
    default:
      return "Unknown illegal instruction";
    }
  case SIGBUS:
    switch (code) {
    case BUS_ADRALN:
      return "Invalid address alignment";
    case BUS_ADRERR:
      return "Non-existent physical address";
    case BUS_OBJERR:
      return "Object specific hardware error";
    default:
      return "Unknown bus error";
    }
  default:
    return "No additional information";
  }
}

static std::string buildSignalInformationString(int sig, siginfo_t *info) {
  std::ostringstream infoStr;
  infoStr << "Signal Information:\n";
  infoStr << "  Signal: " << sig << " (" << getSignalName(sig) << ")\n";
  infoStr << "  Description: " << getSignalDescription(sig) << "\n";

  if (info != nullptr) {
    infoStr << "  Signal Code: " << info->si_code << " ("
            << getSignalCodeDescription(sig, info->si_code) << ")\n";
    infoStr << "  Sending PID: " << info->si_pid << "\n";
    infoStr << "  Sending UID: " << info->si_uid << "\n";

    // Add signal-specific information
    switch (sig) {
    case SIGSEGV:
    case SIGBUS:
      infoStr << "  Fault Address: " << info->si_addr << "\n";
      break;
    case SIGFPE:
    case SIGILL:
      infoStr << "  Fault Address: " << info->si_addr << "\n";
      break;
    case SIGCHLD:
      infoStr << "  Child Status: " << info->si_status << "\n";
      break;
    default:
      if (info->si_addr)
        infoStr << "  Address: " << info->si_addr << "\n";
      break;
    }
  }

  infoStr << "  Process ID: " << getpid() << "\n";
  infoStr << "  Thread ID: " << std::this_thread::get_id() << "\n";

  // Add timestamp
  auto now = std::time(nullptr);
  infoStr << "  Timestamp: " << std::ctime(&now);

  return infoStr.str();
}

static void captureSignalInformation(int sig, siginfo_t *info, void *context) {
  // LLVM's signal handlers give us access to detailed signal information, so
  // capture it, then run LLVM's handlers.
  SignalInfo &sigInfo = signalInformation();
  sigInfo.signal = sig;
  sigInfo.info = buildSignalInformationString(sig, info);

  // Set a timeout to prevent signal handlers from hanging indefinitely.
  // When the heap is corrupted (e.g. SIGABRT from glibc malloc detecting
  // a bad free list), PrintStackTrace may fork llvm-symbolizer which
  // inherits malloc's internal lock and deadlocks. The default SIGALRM
  // action terminates the process, bounding the hang to ~30 seconds.
  alarm(30);

  llvm::sys::RunSignalHandlers();
}

static void logHostMachineInfo(llvm::raw_fd_ostream &crashLog) {
  auto hostMachineOr = M::getHostMachineInfo();
  if (hostMachineOr.isError()) {
    crashLog << "Failed to get machine info. Not including it in the crash log."
             << '\n';
  } else {
    crashLog << "Host machine info:" << '\n';
    M::HostMachineInfo hostInfo = hostMachineOr.takeValue();
    hostInfo.print(crashLog);
  }
}

static void developmentSignalHandler(void *context) {
  std::string *programName = reinterpret_cast<std::string *>(context);

  llvm::errs() << "\n" << std::string(70, '=') << "\n";
  llvm::errs() << *programName << " crashed!\n";
  llvm::errs() << "\n";

  // Output detailed signal information
  const SignalInfo &sigInfo = signalInformation();
  if (!sigInfo.info.empty())
    llvm::errs() << sigInfo.info << "\n";

  llvm::errs() << "C++ stack trace:\n";
  llvm::sys::PrintStackTrace(llvm::errs());
  llvm::errs() << "\n";

  // Store the original signal before potentially triggering SIGUSR2
  int originalSignal = signalInformation().signal;

  // Print Python stack trace if available using async-safe approach
  if (pythonStackTraceEnabled) {
    llvm::errs() << "Python stack trace:\n";
    // Send SIGUSR2 to trigger faulthandler which is async signal safe.
    // The std::raise call will not return until that signal handler
    // is complete (if it is installed from the Python side).
    std::raise(SIGUSR2);
    llvm::errs() << "\n";
  }

  logHostMachineInfo(llvm::errs());

  llvm::errs() << "\n" << std::string(70, '=') << "\n";

  // Use _exit() rather than exit() to avoid re-entering glibc's atexit lock
  // when this handler fires from within an atexit handler context (which
  // exit() holds), and to skip C++ destructors that may be in a broken state.
  _exit(128 + originalSignal);
}

void M::Init::registerDevelopmentSignalHandler(llvm::StringRef programName) {
  // Ensure that the handler is only registered once.
  static std::string programNameStorage;
  static llvm::once_flag flag;
  llvm::call_once(flag, [&]() {
    programNameStorage = std::string(programName);

    // First register with LLVM's signal handling infrastructure
    llvm::sys::AddSignalHandler(developmentSignalHandler,
                                reinterpret_cast<void *>(&programNameStorage));

    // Then install our sigaction handlers after LLVM's handlers are set up.
    // This ensures our handlers get called first, then chain to LLVM's
    // handlers.
    struct sigaction sa;
    sa.sa_sigaction = captureSignalInformation;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);

    // Register sigaction handlers for detailed signal information.
    sigaction(SIGSEGV, &sa, nullptr); // Segmentation fault
    sigaction(SIGABRT, &sa, nullptr); // Abort signal
    sigaction(SIGFPE, &sa, nullptr);  // Floating point exception
    sigaction(SIGILL, &sa, nullptr);  // Illegal instruction
    sigaction(SIGBUS, &sa, nullptr);  // Bus error
    sigaction(SIGTRAP, &sa, nullptr); // Trace/breakpoint trap
    sigaction(SIGSYS, &sa, nullptr);  // Bad system call
  });
}

void M::Init::enablePythonStackTraceCallback() {
  // Enable Python stack traces using async-safe faulthandler mechanism
  // The actual faulthandler registration should be done from Python side
  pythonStackTraceEnabled = true;
}
