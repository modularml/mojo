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

#include "Support/PlatformLibNames.h"
#include "llvm/ADT/StringRef.h"
#include <string>

using namespace M;

std::string PlatformLibrary::getSharedLibraryName(llvm::StringRef name) {
  auto libName =
      std::string(SHARED_LIBRARY_PREFIX) + name.str() + SHARED_LIBRARY_SUFFIX;
  return libName;
}

std::string PlatformLibrary::getStaticLibraryName(llvm::StringRef name) {
  auto libName =
      std::string(STATIC_LIBRARY_PREFIX) + name.str() + STATIC_LIBRARY_SUFFIX;
  return libName;
}
