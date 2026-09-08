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
// This file implements the Tracy bridge for Mojo, providing FFI-callable
// functions for creating and ending Tracy profiling zones.
//
//===----------------------------------------------------------------------===//

#include "Support/SymbolExport.h"
#include <cstddef>
#include <cstdint>

#if MODULAR_KGEN_PROFILING_ENABLED
#include "Support/internal/Tracy/TracyZone.h"
#endif

extern "C" {

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT size_t
KGEN_CompilerRT_TracyIsEnabled(void) {
#ifdef TRACY_ENABLE
  return 1;
#else
  return 0;
#endif
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT uint64_t
KGEN_CompilerRT_TracyZoneBegin(const char *name, size_t nameLen,
                               uint32_t color) {
#if MODULAR_KGEN_PROFILING_ENABLED
  return M::tracyZoneBegin("MojoTrace", sizeof("MojoTrace") - 1,
                           "Trace.__enter__", sizeof("Trace.__enter__") - 1,
                           name, nameLen, color);
#else
  return 0;
#endif
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_TracyZoneEnd(uint64_t packedCtx) {
#if MODULAR_KGEN_PROFILING_ENABLED
  M::tracyZoneEnd(packedCtx);
#endif
}

} // extern "C"
