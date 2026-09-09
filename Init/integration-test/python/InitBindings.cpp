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
#include "Init/Init.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h" // IWYU pragma: keep (type casters)
#include <Python.h>
#include <csignal>
#include <sstream>
#include <stdexcept>

namespace nb = nanobind;

void trigger_segfault_from_cpp() {
  // This will be called from Python, so we'll have Python stack frames
  // available when the signal handler runs
  std::raise(SIGSEGV);
}

void trigger_segfault_with_gil_held() {
  // This function explicitly acquires the GIL in C++ code and then crashes
  // This tests whether our async-safe signal handler can handle crashes
  // that occur while the GIL is held by C++ code

  // Acquire the GIL explicitly
  PyGILState_STATE gstate = PyGILState_Ensure();

  // Trigger a segfault while holding the GIL
  // This is the dangerous scenario that could cause deadlocks with
  // non-async-safe signal handlers
  std::raise(SIGSEGV);

  // This line should never be reached, but for completeness:
  PyGILState_Release(gstate);
}

void initialize_signal_handler(const std::string &program_name) {
  // Initialize the signal handler using the Init module
  auto contextOrError = M::Init::createContext(program_name);
  if (contextOrError.isError()) {
    // Convert Error to string using stringstream
    std::ostringstream oss;
    oss << contextOrError.takeError();
    std::string errorMsg = "Failed to create context: " + oss.str();
    throw std::runtime_error(errorMsg);
  }

  // Enable Python stack traces using async-safe faulthandler mechanism
  // The actual Python stack trace is handled by faulthandler for SIGUSR2
  M::Init::enablePythonStackTraceCallback();

  // No need to store the context for this test - it persists globally
}

NB_MODULE(init_bindings, m) {
  m.doc() = "Init module bindings for testing signal handler with Python stack "
            "traces";

  m.def("trigger_segfault_from_cpp", &trigger_segfault_from_cpp,
        "Trigger a segfault from C++ to test signal handler");

  m.def("trigger_segfault_with_gil_held", &trigger_segfault_with_gil_held,
        "Trigger a segfault from C++ while explicitly holding the GIL");

  m.def(
      "initialize_signal_handler", &initialize_signal_handler,
      "Initialize the development signal handler with the given program name");
}
