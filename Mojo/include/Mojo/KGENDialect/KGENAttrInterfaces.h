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

#ifndef KGEN_KGENDIALECT_KGENATTRINTERFACES_H
#define KGEN_KGENDIALECT_KGENATTRINTERFACES_H

#include "Mojo/KGENDialect/KGENTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"

namespace M::KGEN {
class ParameterEvaluationContext;
class ParameterEvaluator;
class SymTabEvaluationContext;
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "Mojo/KGENDialect/KGENAttrInterfaces.h.inc"

#endif // KGEN_KGENDIALECT_KGENATTRINTERFACES_H
