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

#ifndef KGEN_TRANSFORMUTILS_WALKERS_H
#define KGEN_TRANSFORMUTILS_WALKERS_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace M {
/// Walk the operations contained within operation in reverse, in post order.
/// That means `op` is visited after all the ops in its regions. Ops are visited
/// in reverse order in each region, starting from the last region of each op.
void reversePostOrderWalk(Operation *op,
                          function_ref<void(Operation *)> walkFn);
} // namespace M

#endif // KGEN_TRANSFORMUTILS_WALKERS_H
