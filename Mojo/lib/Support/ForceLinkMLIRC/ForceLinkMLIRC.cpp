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

#include "llvm/Support/raw_ostream.h"

#include "AffineExpr.cpp.inc"
#include "AffineMap.cpp.inc"
#include "BuiltinAttributes.cpp.inc"
#include "BuiltinTypes.cpp.inc"
#include "Diagnostics.cpp.inc"
#include "IR.cpp.inc"
#include "IntegerSet.cpp.inc"
#include "Pass.cpp.inc"
#include "Rewrite.cpp.inc"
#include "Support.cpp.inc"
#include "Transforms.cpp.inc"

namespace M::KGEN {

/// Calling this function forces the linking of MLIR C API symbols. This allows
/// JIT'ed Mojo code to use the same MLIR C API symbols as the current process,
/// which is necessary to avoid conflicting TypeIDs.
void forceLinkMLIRC() {
  forceLinkMLIRCAffineExpr();
  forceLinkMLIRCAffineMap();
  forceLinkMLIRCBuiltinAttributes();
  forceLinkMLIRCBuiltinTypes();
  forceLinkMLIRCDiagnostics();
  forceLinkMLIRCIR();
  forceLinkMLIRCIntegerSet();
  forceLinkMLIRCPass();
  forceLinkMLIRCRewrite();
  forceLinkMLIRCSupport();
  forceLinkMLIRCTransforms();
}

} // namespace M::KGEN
