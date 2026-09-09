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

#ifndef KGEN_SUPPORT_CONSTANTS_H
#define KGEN_SUPPORT_CONSTANTS_H

#include "llvm/ADT/StringRef.h"

namespace M::KGEN {

/// On-disk directory name (relative to the modular cache root) where the
/// Mojo compile cache lives.
inline constexpr llvm::StringLiteral kMojoCacheBaseDirName = ".mojo_cache";

} // namespace M::KGEN

#endif // KGEN_SUPPORT_CONSTANTS_H
