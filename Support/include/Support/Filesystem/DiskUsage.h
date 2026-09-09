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

#ifndef SUPPORT_FILESYSTEM_DISKUSAGE_H
#define SUPPORT_FILESYSTEM_DISKUSAGE_H

#include "Support/ForwardDecls.h"

#include <cstddef>
#include <filesystem>

namespace M {
/// Returns the available disk space in the filesystem containing given path.
ErrorOr<size_t> getAvailableDiskSpace(const std::filesystem::path &path);
} // namespace M

#endif // SUPPORT_FILESYSTEM_DISKUSAGE_H
