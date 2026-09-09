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

#include "Protocol.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "llvm/Support/JSON.h"

namespace json = llvm::json;
namespace KGEN = M::KGEN;
using CO = M::KGEN::CompilationOptions;

namespace M::KGEN::CSP {

//===----------------------------------------------------------------------===//
// KGEN::CompilationOptions
//===----------------------------------------------------------------------===//

bool fromJSON(const json::Value &value, KGEN::CompilationOptions &result,
              json::Path path) {
  json::ObjectMapper o(value, path);
  if (!o)
    return false;

  uint64_t kind;
  if (!o.map("debugLevel", kind))
    return false;
  result.debugLevel = (CO::DebugInfoLevel)kind;

  if (!o.map("debugInfoLanguage", kind))
    return false;
  result.debugInfoLanguage = (CO::DebugInfoLanguage)kind;

  std::optional<uint64_t> optKind;
  if (!o.map("debugAtLevel", optKind))
    return false;

  result.debugAtLevel =
      optKind ? std::make_optional(static_cast<CO::DebugAtLevel>(*optKind))
              : std::nullopt;

  if (!o.map("relocModel", kind))
    return false;
  result.relocModel = (llvm::Reloc::Model)kind;

  return o.map("targetTriple", result.targetTriple) &&
         o.map("targetCpu", result.targetCpu) &&
         o.map("targetFeatures", result.targetFeatures) &&
         o.map("saveTempsPrefix", result.saveTempsPrefix) &&
         o.map("searchPaths", result.searchPaths) &&
         o.map("enableLLVMPerFunctionSplitting",
               result.enableLLVMPerFunctionSplitting) &&
         o.map("enableParallelLLC", result.enableParallelLLC);
}

json::Value toJSON(const KGEN::CompilationOptions &value) {
  return json::Object{
      {"optimizationLevel", value.optimizationLevel},
      {"debugLevel", static_cast<int64_t>(value.debugLevel)},
      {"debugAtLevel", static_cast<std::optional<int64_t>>(value.debugAtLevel)},
      {"targetTriple", value.targetTriple},
      {"targetCpu", value.targetCpu},
      {"targetFeatures", value.targetFeatures},
      {"relocModel", static_cast<int64_t>(value.relocModel)},
      {"debugInfoLanguage", static_cast<int64_t>(value.debugInfoLanguage)},
      {"saveTempsPrefix", value.saveTempsPrefix},
      {"searchPaths", value.searchPaths},
      {"enableLLVMPerFunctionSplitting", value.enableLLVMPerFunctionSplitting},
      {"enableParallelLLC", value.enableParallelLLC}};
}

//===----------------------------------------------------------------------===//
// EmitArchiveParams
//===----------------------------------------------------------------------===//

bool fromJSON(const json::Value &value, EmitArchiveParams &result,
              json::Path path) {
  // Extract compilationOptions field.
  const json::Object *obj = value.getAsObject();
  if (!obj)
    return false;

  const json::Value *fieldValue = obj->get("compilationOptions");
  if (!fieldValue || !fromJSON(*fieldValue, result.compilationOptions, path))
    return false;

  // Extract other fields.
  json::ObjectMapper o(value, path);
  if (!o)
    return false;

  return o.map("module", result.module) && o.map("isJIT", result.isJIT);
}

json::Value toJSON(const EmitArchiveParams &value) {
  return json::Object{{"module", value.module},
                      {"compilationOptions", toJSON(value.compilationOptions)},
                      {"isJIT", value.isJIT}};
}

//===----------------------------------------------------------------------===//
// MLIRModuleParam
//===----------------------------------------------------------------------===//

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, MLIRModule &result,
              llvm::json::Path path) {
  json::ObjectMapper o(value, path);
  return o && o.map("module", result.module);
}

llvm::json::Value toJSON(const MLIRModule &value) {
  return json::Object{{"module", value.module}};
}

//===----------------------------------------------------------------------===//
// ObjectArchive
//===----------------------------------------------------------------------===//

json::Value toJSON(const ObjectArchive &value) {
  return json::Object{{"archive", value.archive}};
}

} // namespace M::KGEN::CSP
