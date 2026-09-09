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

#ifndef SUPPORT_NANOBIND_PYTHONBACKTRACE_H
#define SUPPORT_NANOBIND_PYTHONBACKTRACE_H

#include "nanobind/nanobind.h"

namespace nb = nanobind;

namespace M {

using namespace M;

/// This is a header-only utility to print the Python backtrace from within C++
/// nanobind code. To use this, add `//SDK:Support` to the deps of your
/// particular `modular_nanobind_library` target. This method prints to stdout.
/// It is safe to call this method even if you do not hold the GIL. It will grab
/// it automatically.
/// While you are in lldb, you can also use `_Py_DumpTraceback(1, tstate)`
/// instead.
inline void printPythonBacktrace() {
  nb::gil_scoped_acquire gil;
  auto printStack = nb::module_::import_("traceback").attr("print_stack");
  printStack();
}

} // namespace M

#endif // SUPPORT_NANOBIND_PYTHONBACKTRACE_H
