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

#ifndef KGEN_TRANSFORMUTILS_CONTROLFLOWUTILS_H
#define KGEN_TRANSFORMUTILS_CONTROLFLOWUTILS_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace M::KGEN {
/// Return true if the user of the provided operation is outside the contiguous
/// CFG in which the operation lives. A contiguous CFG is defined as a region
/// subtree where all region operations implement an HLCF interface. Any other
/// operation is assumed to break the CFG, such as inline closures.
///
/// If `enclosingNodes` is non-null, it is appended with the control-flow nodes
/// between `user` and `op`, innermost first. Its contents are meaningless when
/// this returns true, since the walk stops at the operation that breaks the
/// CFG.
bool userCrossesFunctionCFG(Operation *op, Operation *user,
                            SmallVectorImpl<Operation *> *enclosingNodes = {});
} // namespace M::KGEN

#endif // KGEN_TRANSFORMUTILS_CONTROLFLOWUTILS_H
