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

#include "Support/Filesystem/DiskUsage.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/Twine.h"
#include <cstddef>
#include <filesystem>
#include <system_error>

using namespace M;

M::ErrorOr<size_t> M::getAvailableDiskSpace(const std::filesystem::path &path) {
  std::error_code ec;
  std::filesystem::space_info info = std::filesystem::space(path, ec);
  if (ec)
    return Error(llvm::Twine(ec.message()));
  return info.available;
}
