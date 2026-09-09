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
// This file implements structs for Compile Server Protocol.
//
// Each struct has a toJSON and fromJSON function, that converts between
// the struct and a JSON representation. (See JSON.h)
//
// Some structs also have operator<< serialization. This is for debugging and
// tests, and is not generally machine-readable.

//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_CS_PROTOCOL_H
#define KGEN_TOOLS_CS_PROTOCOL_H

#include "Mojo/ToolCommon/CompilationOptions.h"
#include "llvm/Support/JSON.h"
#include <string>

namespace KGEN = M::KGEN;
namespace M::KGEN::CSP {

//===----------------------------------------------------------------------===//
// KGEN::CompilationOptions
//===----------------------------------------------------------------------===//

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, KGEN::CompilationOptions &result,
              llvm::json::Path path);
llvm::json::Value toJSON(const KGEN::CompilationOptions &value);

//===----------------------------------------------------------------------===//
// EmitArchiveParams
//===----------------------------------------------------------------------===//

struct EmitArchiveParams {
  /// MLIR module printed as string.
  std::string module;

  /// Compilation options.
  KGEN::CompilationOptions compilationOptions;

  /// Indicates if the code should be JITted.
  bool isJIT;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, EmitArchiveParams &result,
              llvm::json::Path path);
llvm::json::Value toJSON(const EmitArchiveParams &value);

//===----------------------------------------------------------------------===//
// MLIRModuleParam
//===----------------------------------------------------------------------===//

struct MLIRModule {
  /// MLIR module printed as string.
  std::string module;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, MLIRModule &result,
              llvm::json::Path path);
llvm::json::Value toJSON(const MLIRModule &value);

//===----------------------------------------------------------------------===//
// ObjectArchive
//===----------------------------------------------------------------------===//

/// Represents an object archive obtained as a result
/// of the compilation.
struct ObjectArchive {
  /// Object archive encoded as a string.
  std::string archive;
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const ObjectArchive &value);

} // namespace M::KGEN::CSP

#endif // KGEN_TOOLS_CS_PROTOCOL_H
