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
// This file contains various utilities for interacting with the Mojo REPL.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOTOOLING_REPLPYTHONEXPRUTILS_H
#define KGEN_MOJOTOOLING_REPLPYTHONEXPRUTILS_H

#include "Support/ErrorOr.h"
#include "llvm/ADT/StringRef.h"
#include <vector>

namespace M {
namespace KGEN {
namespace Mojo {

/// This class represents an abstract extracted python symbol.
class ExtractedPythonSymbol {
public:
  enum class Kind {
    /// A python import statement.
    Import,
    /// A python declaration statement.
    Decl,
  };

  /// Return the exposed name of this decl.
  StringRef getName() const { return name; }

  /// Return the kind of this decl.
  Kind getKind() const { return kind; }

protected:
  ExtractedPythonSymbol(Kind kind, StringRef name)
      : kind(kind), name(name.str()) {}

private:
  Kind kind;
  std::string name;
};

/// This class represents an extracted python import statement.
class ExtractedPythonImport : public ExtractedPythonSymbol {
public:
  ExtractedPythonImport(StringRef name, StringRef module)
      : ExtractedPythonSymbol(Kind::Import, name), module(module.str()) {}

  /// Return the module name of this import.
  StringRef getModule() const { return module; }

  /// Support llvm-style casting.
  static bool classof(const ExtractedPythonSymbol *symbol) {
    return symbol->getKind() == Kind::Import;
  }

private:
  std::string module;
};

/// This class represents an extracted python declaration statement.
class ExtractedPythonDecl : public ExtractedPythonSymbol {
public:
  ExtractedPythonDecl(StringRef name)
      : ExtractedPythonSymbol(Kind::Decl, name) {}

  /// Support llvm-style casting.
  static bool classof(const ExtractedPythonSymbol *symbol) {
    return symbol->getKind() == Kind::Decl;
  }
};

/// Extract the importable python symbols from the given python expression
/// string.
ErrorOr<std::vector<std::unique_ptr<ExtractedPythonSymbol>>>
extractPythonSymbolsFromReplExpr(StringRef pythonExpr);

} // namespace Mojo
} // namespace KGEN
} // namespace M

#endif // KGEN_MOJOTOOLING_REPLPYTHONEXPRUTILS_H
