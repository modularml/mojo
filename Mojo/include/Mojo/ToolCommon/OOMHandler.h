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

#ifndef KGEN_TOOLCOMMON_OOMHANDLER_H
#define KGEN_TOOLCOMMON_OOMHANDLER_H

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

namespace M::KGEN {

/// Install a user-friendly out-of-memory error handler so that allocation
/// failures produce an actionable message instead of a crash.
///
/// Call this once, early in main(), after llvm::InitLLVM.
inline void installOOMHandler() {
  llvm::install_bad_alloc_error_handler([](void *, const char *, bool) {
    llvm::errs()
        << "error: the Mojo compiler ran out of memory.\n"
           "If your program has excessive compile-time computations or "
           "comptime recursion, try simplifying it.\n"
           "If that does not help, this may be a Mojo compiler bug; please "
           "file a report at https://github.com/modular/modular/issues\n";
    exit(1);
  });
}

} // namespace M::KGEN

#endif // KGEN_TOOLCOMMON_OOMHANDLER_H
