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
#ifndef KGEN_HLCFDIALECT_HLCFATTRS_H
#define KGEN_HLCFDIALECT_HLCFATTRS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"

//===----------------------------------------------------------------------===//
// UnrollLevel
//===----------------------------------------------------------------------===//

namespace M::HLCF {
/// This class describes the unroll level of a loop. If the value is greater
/// than zero, then it is interpreted as the unroll factor of the loop. If it is
/// 0, the loop is not unrolled. If set to -1, the loop is fully unrolled.
class UnrollLevel {
public:
  UnrollLevel(int32_t value) : value(value) {}

  static UnrollLevel none() { return 0; }
  static UnrollLevel full() { return -1; }

  bool isFull() const { return *this == full(); }
  bool isNone() const { return *this == none(); }
  bool isFactor() const { return value > 0; }
  int32_t getFactor() const { return value; }

  bool operator==(UnrollLevel other) const { return value == other.value; }

  /// Enable hashing for integration with `UnrollLevelAttr`.
  llvm::hash_code hash() const;

private:
  /// The unroll level encoding.
  int32_t value;
};
} // namespace M::HLCF

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "Mojo/HLCFDialect/HLCFEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Mojo/HLCFDialect/HLCFAttrs.h.inc"

#endif // KGEN_HLCFDIALECT_HLCFATTRS_H
