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
// This file contains helpers to write MLIR op verifiers.
//
//===----------------------------------------------------------------------===//

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"

namespace M {

/// Check that the op has the expected operand types.
LogicalResult checkOperandTypes(Operation *op, TypeRange expectedTypes);

/// Check that the block with the given name has the expected argument types.
/// The given op is only used to emit any errors.
LogicalResult checkArgumentTypes(Operation *op, StringRef blockName,
                                 Block *block, TypeRange expectedTypes);

/// Check that the two type ranges with the given context label agree. The given
/// op is only used to emit any errors.
LogicalResult checkMatchingTypes(Operation *op, StringRef context,
                                 TypeRange actualTypes,
                                 TypeRange expectedTypes);

} // namespace M
