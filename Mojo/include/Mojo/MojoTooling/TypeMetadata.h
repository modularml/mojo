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

#ifndef KGEN_MOJOTOOLING_TYPEMETADATA_H
#define KGEN_MOJOTOOLING_TYPEMETADATA_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include <string>

namespace M {
namespace KGEN {

//===----------------------------------------------------------------------===//
// TypeMetadata
//===----------------------------------------------------------------------===//

/// Information about a type used to generate doc details in JSON
class TypeMetadata {
public:
  TypeMetadata() = default;
  TypeMetadata(llvm::StringRef typeStr, llvm::StringRef module = "",
               llvm::StringRef relativePath = "")
      : typeString(typeStr.str()), moduleNamespace(module.str()),
        relativeDocPath(relativePath.str()) {}

  /// Get the module namespace (e.g., "builtin.int", "collections.list")
  llvm::StringRef getModuleNamespace() const { return moduleNamespace; }

  /// Get the relative documentation path for cross-references
  llvm::StringRef getRelativeDocPath() const { return relativeDocPath; }

  /// Serialize the metadata to JSON with the following schema:
  /// {
  ///   "type": string,           // Full type as written in source, including
  ///                             // any parameterization (e.g. "List[Int]").
  ///   "path": string,           // Relative documentation path for the base
  ///                             // type's doc page (optional).
  /// }
  llvm::json::Object toJSON() const;

private:
  std::string typeString;
  std::string moduleNamespace; // Module namespace: "builtin.int", etc.
  std::string relativeDocPath; // Relative path for cross-reference links
};

} // namespace KGEN
} // namespace M

#endif // KGEN_MOJOTOOLING_TYPEMETADATA_H
