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

#ifndef KGEN_CODIALECT_COTYPES_H
#define KGEN_CODIALECT_COTYPES_H

#include "Support/LLVMForwardDecls.h"
#include "Support/MDialect/MTypeInterfaces.h"
#include "mlir/IR/Types.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Mojo/CODialect/COTypes.h.inc"

#endif // KGEN_CODIALECT_COTYPES_H
