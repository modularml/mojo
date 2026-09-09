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

#ifndef SUPPORT_FILESYSTEM_PATHS_H
#define SUPPORT_FILESYSTEM_PATHS_H

#include <filesystem>

namespace M::Filesystem {

/// Returns true if the given path is a Mojo package source directory (i.e. a
/// directory that contains an `__init__.mojo` file).
bool isMojoSourcePackagePath(const std::filesystem::path &path);

/// Returns true if the given path is a Mojo binary package (i.e. a
/// `.mojoc` or `.mojopkg` file).
bool isMojoBinaryPackagePath(const std::filesystem::path &path);

/// Return if the given file path defines a mojo source file.
bool isMojoSourceFile(const std::filesystem::path &path);

/// Return if the given file path defines a MLIR bytecode file (`.mlirbc`).
bool isMLIRByteCodeFile(const std::filesystem::path &path);

} // namespace M::Filesystem

#endif // SUPPORT_FILESYSTEM_PATHS_H
