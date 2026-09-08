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

#include "Support/Process.h"
#include "Support/SymbolExport.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include <cctype>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace M;

using llvm::sys::findProgramByName;
using llvm::sys::fs::is_regular_file;

// Works across ubuntu 20.04, 22.04, macos, pyenv, conda, venv, virtual
const char *FIND_LIBPYTHON = R"PROG(
import os
import sys
from pathlib import Path
from sysconfig import get_config_var
ext = "dll" if os.name == "nt" else "dylib" if sys.platform == "darwin" else "so"
pyver = get_config_var("py_version_short")
abiflags = get_config_var("ABIFLAGS") or ""
binary = f"libpython{pyver}{abiflags}.{ext}"
for libpython in [Path(get_config_var(p)) / binary for p in ["LIBPL", "LIBDIR"]]:
    if libpython.exists():
        print(libpython.resolve())
        exit(0)
exit(1)
)PROG";

//===----------------------------------------------------------------------===//
// KGEN_CompilerRT_Python_SetPythonPath
//===----------------------------------------------------------------------===//

// TODO: add a subprocess module to Mojo so this can all be done natively
// Returns path to a libpython of the same version as `pythonBin`
static std::optional<std::string> findLibPython(const std::string &pythonBin) {
  std::string cmd = pythonBin + " -c '" + FIND_LIBPYTHON + "'";
  std::array<char, 128> buffer;
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"),
                                                pclose);
  if (!pipe) {
    return std::nullopt;
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  std::erase_if(result, ::isspace);
  return result;
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT const char *
KGEN_CompilerRT_Python_SetPythonPath() {
  auto pythonBin = llvm::sys::Process::GetEnv("MOJO_PYTHON").value_or("");
  if (!pythonBin.empty() && !is_regular_file(pythonBin))
    return "`MOJO_PYTHON` is not set to a file path.";

  // If `MOJO_PYTHON` is not found, use python3 or python from  on top of `PATH`
  if (pythonBin.empty()) {
    auto pythonBinErr = findProgramByName("python3");
    if (!pythonBinErr)
      pythonBinErr = findProgramByName("python");
    if (pythonBinErr)
      pythonBin = *pythonBinErr;
  }

  // `PYTHONEXECUTABLE` enables multiprocessing, and adding virtual environment
  // site-modules. Not strictly required in an environment with no executable.
  if (!pythonBin.empty())
    if (failed(setProcessEnv("PYTHONEXECUTABLE", pythonBin)))
      return "cannot set `PYTHONEXECUTABLE` to";

  // If `MOJO_PYTHON_LIBRARY` is not set, run a Python script to find it.
  auto libpython = llvm::sys::Process::GetEnv("MOJO_PYTHON_LIBRARY");
  if (!libpython && !pythonBin.empty())
    libpython = findLibPython(pythonBin);

  // Intentionally setting MOJO_PYTHON_LIBRARY to "" should result in
  // `dlopen(nullptr, ..)`, to look for CPython symbols in the current process.
  // That behavior is important on platforms (Linux), where the Python `python`
  // executable statically links the CPython implementation but can't be
  // `dlopen()`'d directly because it is a PIE executable.
  if (!libpython || (*libpython != "" && !is_regular_file(*libpython)))
    return "found no suitable Python library to link to";

  if (failed(setProcessEnv("MOJO_PYTHON_LIBRARY", *libpython)))
    return "cannot set `MOJO_PYTHON_LIBRARY`";

  return "";
}
