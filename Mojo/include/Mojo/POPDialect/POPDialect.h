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
// This file defines the POP MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_POPDIALECT_H
#define KGEN_POPDIALECT_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Dialect.h"

// Pull in the dialect definition.
#include "Mojo/POPDialect/POPDialect.h.inc"

#endif // KGEN_POPDIALECT_H
