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

#ifndef KGEN_TRANSFORMUTILS_SLICINGUTILS_H
#define KGEN_TRANSFORMUTILS_SLICINGUTILS_H

#include "Mojo/KGENDialect/KGENUtils.h"
#include "Support/LLVMCompilerForwardDecls.h"

namespace M {
/// Slice the dependencies of an operation out of the existing module into the
/// self-contained slice module.
void sliceDependencies(Operation *op, SymbolTable &sliceSymtab,
                       const SymbolTable &symtab, IRMapping &reusedMapping,
                       DenseSet<const void *> &visited);

/// Produce a standalone MLIR module by slicing out the dependencies of the
/// provided exported ops.
OwningOpRef<ModuleOp>
produceStandaloneModule(const SymbolTable &symtab,
                        const KGEN::ExportMap &exportedSymbols,
                        bool overrideExported = false);

/// Produce a standalone MLIR module by slicing out the dependencies of the
/// provided exported ops. An `IRMapping` can be provided to be able to map
/// into the sliced module.
OwningOpRef<ModuleOp>
produceStandaloneModule(const SymbolTable &symtab,
                        const KGEN::ExportMap &exportedSymbols,
                        IRMapping &mapping, bool overrideExported = false);
} // namespace M

#endif // KGEN_TRANSFORMUTILS_SLICINGUTILS_H
