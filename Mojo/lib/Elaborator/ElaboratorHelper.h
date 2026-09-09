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

#ifndef KGEN_ELABORATOR_ELABORATORHELPER_H
#define KGEN_ELABORATOR_ELABORATORHELPER_H

#include "Mojo/KGENDialect/KGENOps.h"
#include "Support/Compiler/ErrorTree.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SetVector.h"

namespace M::KGEN {

/// Compute the final symbol name for a function decorated with an explicit
/// linkage name. This is the single source of truth shared by both
/// renameFunctions (GPU rename loop) and applyLinkageName (host-side
/// get_linkage_name evaluation).
///
/// \param resolved  The @__name string value (verbatim, unsanitized).
/// \param lna       The linkageName attribute carrying the mangle flag.
/// \param sanitize  Whether or not to sanitize linkage names. See
///                  sanitizeSymbolToUnderscores for the sanitization scheme.
/// \param symName   The auto-mangled symbol name for hashing (mangle=true
///                  only): for non-parametric kernels this is the linkage name
///                  literal; for parametric kernels it's mangleParameterValues.
/// \param funcType  The MLIR function type, used as a hash input (mangle=true).
mlir::StringAttr applyLinkageName(mlir::StringAttr resolved,
                                  LinkageNameAttr lna, bool sanitize,
                                  llvm::StringRef symName,
                                  mlir::FunctionType funcType);

/// Resolve linkage names and sanitize symbol names for all FuncOps in
/// `theModule` in a single pass:
///
///  1. If a FuncOp carries a `linkageName` attribute, compute the final symbol
///     name (sanitized on GPU, verbatim on host), then remove the attribute.
///  2. On GPU targets, additionally sanitize every other name to
///     alphanumeric-only characters for PTX compatibility.
///  3. Fix up all `SymbolConstantAttr` references in the module to reflect the
///     new names.
///
/// Sets `failed = true` on errors (unresolved linkage name, duplicate symbols).
void renameFunctions(mlir::ModuleOp theModule, bool isGPU, bool &failed);

/// Bundle a set of `compile_offload` ops in a deterministic order that does not
/// depend on the order elaboration discovered them (which varies run to run,
/// since `offloadOps` is populated concurrently during parallel elaboration).
/// Ops are sorted by their mangled kernel name, tie-broken on the target and
/// emission attributes, then `bundleOp` is invoked for each op with its mangled
/// name.
///
/// \param offloadOps  Ops to bundle; their generators must live in `symTab`.
/// \param symTab      Symbol table used to resolve each op's `func` attribute.
/// \param bundleOp    Per-elaborator bundling of one op given its mangled name.
ErrorTreeOrSuccess sortAndBundleOffloadOps(
    const llvm::SetVector<CompileOffloadOp> &offloadOps,
    mlir::SymbolTable &symTab,
    llvm::function_ref<ErrorTreeOrSuccess(CompileOffloadOp, mlir::StringAttr)>
        bundleOp);

} // namespace M::KGEN

#endif // KGEN_ELABORATOR_ELABORATORHELPER_H
