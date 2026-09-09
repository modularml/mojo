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

#include "Mojo/MojoTooling/TypeExtractionUtils.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/ASTType.h"
#include "Mojo/MojoParser/SharedState.h"
#include "Mojo/MojoParser/SignatureModel.h"
#include "Mojo/MojoTooling/PublicASTDecl.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <map>
#include <mutex>

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

namespace M {
namespace KGEN {

namespace TypeExtractionUtils {

/// Extracts the leaf name from a symbol reference.
std::string extractSymbolLeafName(mlir::SymbolRefAttr symbol) {
  if (symbol.getNestedReferences().empty()) {
    return symbol.getRootReference().getValue().str();
  }
  return symbol.getNestedReferences().back().getAttr().getValue().str();
}

/// Gets the base type name, removing generic parameters and qualifiers.
std::string extractBaseTypeName(const M::MojoASTTypeRef &astType,
                                llvm::StringRef fullTypeStr) {
  // Try AST-based extraction first
  if (astType) {
    Type mlirType = astType.getMLIRType();

    // Handle struct types
    if (auto structType = sugarDynCast<LIT::StructType>(mlirType)) {
      if (auto symbol = structType.getSymbol()) {
        return extractSymbolLeafName(symbol);
      }
    }
    // Handle reference types
    else if (auto refType = sugarDynCast<LIT::RefType>(mlirType)) {
      if (auto elementType = astType.getReferenceElementType()) {
        return extractBaseTypeName(elementType, fullTypeStr);
      }
    }
    // Handle trait types
    else if (auto traitType = sugarDynCast<LIT::TraitType>(mlirType)) {
      auto symbols = traitType.getSymbols();
      if (!symbols.empty()) {
        return extractSymbolLeafName(symbols.front().getSymbol());
      }
    }
    // Handle OriginSet type name
    else if (sugarIsa<LIT::OriginSetType>(mlirType)) {
      return "OriginSet";
    }
  }

  // String-based fallback: remove generics
  llvm::StringRef baseType = fullTypeStr;

  // Special case if not found in AST:
  // If the string looks like a closure function signature, keep it as-is
  if (baseType.starts_with("fn(") || baseType.starts_with("fn[") ||
      baseType.starts_with("def(") || baseType.starts_with("def[")) {
    return baseType.trim().str();
  }

  if (size_t bracketPos = baseType.find('[');
      bracketPos != llvm::StringRef::npos) {
    baseType = baseType.substr(0, bracketPos);
  }
  baseType = baseType.trim();
  return baseType.str();
}

/// Convenience for extracting base type names when no AST info is available.
std::string extractBaseTypeName(llvm::StringRef fullTypeStr) {
  return extractBaseTypeName(M::MojoASTTypeRef{}, fullTypeStr);
}

/// Extracts the fully qualified module path from an AST declaration reference.
std::string extractModulePathFromDecl(M::MojoASTDeclRef declRef) {
  llvm::SmallVector<llvm::StringRef> pathComponents;

  M::MojoASTDeclRef current = declRef;
  while (current) {
    Operation *op = current.getIfOperation();
    if (auto fileModule = dyn_cast_or_null<FileModuleOp>(op)) {
      // File module - add its name and continue up to get full hierarchy
      if (auto name = current.getName()) {
        pathComponents.push_back(*name);
      }
    } else if (auto packageOp = dyn_cast_or_null<PackageOp>(op)) {
      // Package - add its name and continue up
      if (auto name = current.getName()) {
        pathComponents.push_back(*name);
      }
    }
    current = current.getParent();
  }

  std::reverse(pathComponents.begin(), pathComponents.end());

  std::string result;
  for (size_t i = 0; i < pathComponents.size(); ++i) {
    if (i > 0)
      result += ".";
    result += pathComponents[i].str();
  }

  return result;
}

/// Attempts to resolve a type name (eg, "List") to its actual AST declaration.
std::optional<M::MojoASTDeclRef>
tryResolveTypeToDecl(llvm::StringRef typeName, SharedState &sharedState,
                     const M::MojoASTDeclRef *contextDecl) {
  // Strategy 1: If we have a context declaration, try looking up the type in
  // its scope and parent scopes
  if (contextDecl && *contextDecl) {
    M::MojoASTDeclRef current = *contextDecl;
    while (current) {
      // Look for the type in the current scope
      for (auto childEntry : current.getChildren()) {
        if (childEntry.getName() == typeName) {
          for (auto child : childEntry.getDecls()) {
            Operation *childOp = child.getIfOperation();
            if (isa_and_nonnull<StructDeclOp, TraitDeclOp, AliasDeclOp>(
                    childOp)) {
              return child;
            }
          }
        }
      }
      current = current.getParent();
    }
  }

  // Strategy 2: Try looking up builtin traits using SharedState
  if (contextDecl && *contextDecl) {
    if (ASTDecl *builtinTraitDecl =
            sharedState.lookupBuiltinTrait(typeName, SMLoc())) {
      return M::MojoASTDeclRef(builtinTraitDecl);
    }
  }

  return std::nullopt;
}

/// Generates a documentation path from module info for cross-linking.
std::string generateDocPath(llvm::StringRef module, llvm::StringRef typeName,
                            llvm::StringRef docsBasePath, bool isAlias) {
  if (typeName.empty()) {
    return "";
  }

  // Remove __init__ components from the module path since they shouldn't appear
  // in documentation URLs (APIs defined in __init__.mojo files should link to
  // the parent package/module)
  llvm::SmallVector<llvm::StringRef> components;
  module.split(components, '.');

  std::string moduleStr;
  for (size_t i = 0; i < components.size(); ++i) {
    if (components[i] != "__init__") {
      if (!moduleStr.empty()) {
        moduleStr += ".";
      }
      moduleStr += components[i].str();
    }
  }

  std::string path;
  bool isStdType = moduleStr.starts_with("std.");

  // If this is a std type, use the module path as-is
  if (isStdType) {
    path = moduleStr;
    // Add underscore at the end of ".index" name for docsite URL compatibility
    if (path.length() >= 6 && path.substr(path.length() - 6) == ".index") {
      path = path.substr(0, path.length() - 6) + ".index_";
    }
  } else {
    if (!docsBasePath.empty()) {
      path = docsBasePath.str();
      if (!moduleStr.empty()) {
        path += "/" + moduleStr;
      }
    } else {
      path = moduleStr;
    }
  }

  std::replace(path.begin(), path.end(), '.', '/');
  path = "/" + path;

  if (isAlias) {
    // For aliases, add the fragment identifier with lowercase alias name
    std::string aliasStr = typeName.str();
    std::transform(aliasStr.begin(), aliasStr.end(), aliasStr.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (!path.empty()) {
      path += "/#" + aliasStr;
    } else {
      path = "#" + aliasStr;
    }
  } else {
    // For concrete types, append the type name normally
    if (!path.empty()) {
      path += "/" + typeName.str();
    } else {
      path = typeName.str();
    }
  }

  return path;
}

/// Resolved cross-reference info for a base type name. The display string
/// (`"type"`) is the caller's full parameterized type and is not cached here,
/// so different parameterizations of the same base type (e.g. `List[Int]` vs
/// `List[String]`) share the same path resolution without overwriting each
/// other's display string.
///
/// An entry with both fields empty is a negative-cache marker — used only when
/// `extractLibraryInfo` is called without `SharedState`, since AST resolution
/// may succeed later given better context.
namespace {
struct ResolvedPath {
  std::string moduleNamespace;
  std::string docPath;
};
} // namespace

/// The main function that extracts comprehensive type metadata from type names.
TypeMetadata extractLibraryInfo(llvm::StringRef typeStr,
                                const M::MojoASTDeclRef *currentDeclContext,
                                SharedState *sharedState) {
  // Cache for resolved cross-reference paths, keyed by stripped base name.
  static std::map<std::string, ResolvedPath> pathCache;
  static std::mutex cacheMutex;

  auto makeMetadata = [&](const ResolvedPath &resolved) {
    return TypeMetadata(typeStr, resolved.moduleNamespace, resolved.docPath);
  };

  // Extract base type name: remove templates and qualified prefixes
  llvm::StringRef baseType = typeStr;

  // Remove template parameters (everything after '[')
  if (size_t bracketPos = baseType.find('[');
      bracketPos != llvm::StringRef::npos) {
    baseType = baseType.substr(0, bracketPos);
  }

  baseType = baseType.trim();

  // Check cache first (with thread safety).
  std::string cacheKey = baseType.str();
  {
    std::lock_guard<std::mutex> lock(cacheMutex);
    if (auto cachedIt = pathCache.find(cacheKey); cachedIt != pathCache.end()) {
      return makeMetadata(cachedIt->second);
    }
  }

  // Try AST-based resolution if we have shared state
  if (sharedState) {
    if (auto declRef =
            tryResolveTypeToDecl(baseType, *sharedState, currentDeclContext)) {
      std::string modulePath = extractModulePathFromDecl(*declRef);
      if (!modulePath.empty()) {
        std::string typeName;
        std::string docPath;

        // Check if this is an alias declaration
        bool isAlias = isa_and_nonnull<AliasDeclOp>(declRef->getIfOperation());
        if (isAlias) {
          // Preserve the original alias name (used to build the URL anchor)
          typeName = baseType.str();
        } else {
          // For concrete types (structs/traits), use the regular path
          typeName = extractBaseTypeName(typeStr);
        }
        docPath = generateDocPath(modulePath, typeName,
                                  sharedState->getDocsBasePath(), isAlias);

        ResolvedPath resolved{modulePath, docPath};
        {
          std::lock_guard<std::mutex> lock(cacheMutex);
          pathCache[cacheKey] = resolved;
        }
        return makeMetadata(resolved);
      }
    }
  }

  // For unknown types (not found in AST), return minimal info with no path.
  // Don't cache failures when we have SharedState - they might succeed later
  // with better context.
  //
  // Asymmetry: the lookup at the top of the function does *not* re-check
  // `sharedState`, so a negative entry inserted here will be returned on
  // subsequent calls even if a later caller has SharedState. In practice the
  // same baseType is queried with consistent SharedState availability, so
  // this hasn't bitten us; revisit if it ever does.
  if (!sharedState) {
    std::lock_guard<std::mutex> lock(cacheMutex);
    pathCache[cacheKey] = ResolvedPath{};
  }
  return makeMetadata(ResolvedPath{});
}

std::string stripImplicitArgParams(llvm::StringRef typeStr,
                                   llvm::StringRef argName) {
  return ::stripImplicitArgParams(typeStr, argName);
}

} // namespace TypeExtractionUtils
} // namespace KGEN
} // namespace M
