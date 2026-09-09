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

#ifndef KGEN_TOOL_COMMON_DEBUG_H
#define KGEN_TOOL_COMMON_DEBUG_H

namespace M::KGEN {

#ifndef NDEBUG
bool isCurrentDebugTypeLevel(const char *Type, unsigned level);

/// KGEN_DEBUG_WITH_TYPE_LEVEL macro - This macro should be used by passes to
/// emit debug information.  If the '-kgen-debug-only' option is specified on
/// the commandline, and if this is a debug build, and debug type is listed in
/// the option, and LEVEL is smaller than the maximul level specified for the
/// debug type,then the code specified as the option to the macro will be
/// executed.  Otherwise it will not be. Example:
///
/// -kgen-debug-only=elaborator:5
/// or
/// -kgen-debug-only=elaborator
///
/// KGEN_DEBUG_WITH_TYPE_LEVEL("elaborator", 5, dbgs() << "elab: " << generator
/// << "\n");
///
/// The code above won't be executed in case if
///
/// -kgen-debug-only=elab
/// or
/// -kgen-debug-only=elabator:3
///
/// That is, LEVEL is essential to reduce the number of debug messages, but it's
/// crucial to place vital information of the pass at the lower LEVEL.
///
#define KGEN_DEBUG_WITH_TYPE_LEVEL(TYPE, LEVEL, ...)                           \
  do {                                                                         \
    if (::M::KGEN::debugFlag &&                                                \
        ::M::KGEN::isCurrentDebugTypeLevel(TYPE, LEVEL)) {                     \
      __VA_ARGS__;                                                             \
    }                                                                          \
  } while (false)

#else

#define isCurrentDebugTypeLevel(X, Y) (false)
#define KGEN_DEBUG_WITH_TYPE_LEVEL(TYPE, LEVEL, ...)                           \
  do {                                                                         \
  } while (false)

#endif // NDEBUG

#define KGEN_DEBUG(LEVEL, ...)                                                 \
  KGEN_DEBUG_WITH_TYPE_LEVEL(KGEN_DEBUG_TYPE, LEVEL, __VA_ARGS__)

extern bool debugFlag;

void initializeDebugOptions();

} // namespace M::KGEN
#endif // KGEN_TOOL_COMMON_DEBUG_H
