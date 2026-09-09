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

#ifndef SUPPORT_CPUCACHE_H
#define SUPPORT_CPUCACHE_H

#include "Support/ForwardDecls.h"
#include <cstddef>

namespace M {
//===----------------------------------------------------------------------===//
// Cache sizes
//===----------------------------------------------------------------------===//

/// Get the D$ or unified cache size in bytes at a 1-based cache level index.
/// An error is returned if there is an OS error in finding the cache level.  If
/// the cache level does not exist, 0 is returned.
ErrorOr<size_t> getHostCPUCacheSize(size_t cacheLevel);
} // namespace M

#endif // SUPPORT_CPUCACHE_H
