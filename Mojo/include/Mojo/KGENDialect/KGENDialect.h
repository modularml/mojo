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
// This file defines an KGEN MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_H
#define KGEN_KGENDIALECT_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/IR/Dialect.h"
#include "llvm/ADT/StringMap.h"

#include <filesystem>

// Pull in the dialect definition.
#include "Mojo/KGENDialect/KGENDialect.h.inc"

namespace M::KGEN {
/// Register KGEN dialect commandline options.
void registerKGENCommandLineOptions();

template <typename TypeT>
void KGENDialect::registerMnemonicType() {
  registerPrettyType(
      TypeT::getMnemonic(), +[](AsmParser &p) { return TypeT::parse(p); },
      mlir::TypeID::get<TypeT>(),
      +[](AsmPrinter &p, Type type) {
        p << TypeT::getMnemonic();
        cast<TypeT>(type).print(p);
      });
}
} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_H
