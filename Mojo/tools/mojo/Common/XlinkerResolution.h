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
// Helpers for translating user-supplied `-Xlinker` arguments into a list of
// shared libraries that an in-process JIT (e.g. `mojo run`, `mojo debug`)
// should `dlopen()`. Shared between the JIT entry points so each one does not
// reimplement the `ld`-style `-L`/`-l`/`-rpath` resolution.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_XLINKERRESOLUTION_H
#define KGEN_TOOLS_MOJO_COMMON_XLINKERRESOLUTION_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace M {
struct State;

/// Translates user-supplied `-Xlinker` arg values into a list of absolute
/// paths to shared libraries that an in-process JIT should `dlopen()`. JIT
/// modes have no native linker — the program is loaded in-process via ORC —
/// so we emulate the subset of `ld` semantics that maps `-L<dir>` /
/// `-l<name>` / absolute library paths to a set of libraries the JIT can
/// resolve symbols from.
///
/// Recognized forms:
///   - `-L<dir>` or `-L <dir>`: add `<dir>` to the library search path.
///   - `--library-path=<dir>` or `--library-path <dir>`: same as `-L`.
///   - `-l<name>` or `-l <name>`: locate `lib<name>.so` (Linux) or
///     `lib<name>.dylib` (macOS) within the search path.
///   - `-rpath <dir>` / `-rpath=<dir>` (also `--rpath`): conflated with
///     `-L`. Strictly, `-rpath` only affects the runtime loader's
///     `DT_NEEDED` resolution, not `-l<name>` lookup; but under JIT the
///     linker and runtime loader collapse into a single dlopen pass, so
///     treating it as another search dir matches what users expect.
///   - An existing path whose filename has a shared-library extension
///     (`.so`, `.dylib`, or a versioned `.so.N` form): loaded directly.
///
/// Unrecognized forms (and `-l<name>` references that fail to resolve) are
/// reported as non-fatal warnings and dropped. We deliberately diverge from
/// `ld`'s strict-error behavior: under JIT, symbols may still resolve
/// against the host process or libraries already loaded by the runtime, so
/// erroring on a missing `-l` would be more aggressive than the JIT-mode
/// model warrants. If a needed symbol truly is missing, the JIT will fail
/// at lookup time with a symbol-not-found diagnostic.
///
/// We deliberately do not add default search paths (e.g. `/usr/lib`,
/// `LD_LIBRARY_PATH`); `mojo build` doesn't add any either — it forwards
/// `-Xlinker` to the system linker, which has its own builtin defaults.
/// Symbols in libraries already loaded by the host process (libc, libm,
/// etc.) resolve through ORC's process-symbol generator without any
/// explicit `-l`.
llvm::SmallVector<std::string>
resolveXlinkerLibraries(const State &state,
                        llvm::ArrayRef<std::string> xlinkerArgs);

} // namespace M

#endif // KGEN_TOOLS_MOJO_COMMON_XLINKERRESOLUTION_H
