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

#include "Mojo/KGENDialect/KGENDType.h"
#include "Support/Compiler/MLIRDType.h"

using namespace M;
using namespace KGEN;

/// Return the element type for it's string representation.
FailureOr<KGENDType> KGENDType::getFromString(StringRef str) {
  if (str == "address")
    return KGENDType(ExtraCases::address);
  if (str == "index")
    return KGENDType(ExtraCases::index);
  if (str == "uindex")
    return KGENDType(ExtraCases::uindex);
  auto dtype = DType::getFromString(str);
  if (succeeded(dtype))
    return KGENDType(*dtype);
  return failure();
}

/// Return a string form of this DType value, following the library spelling.
static std::string getAsLongString(uint8_t dtype) {
  switch (dtype) {
  case KGENDType::ui1:
    return "_uint1";
  case KGENDType::ui2:
    return "_uint2";
  case KGENDType::ui4:
    return "_uint4";
  case KGENDType::ui8:
    return "uint8";
  case KGENDType::si8:
    return "int8";
  case KGENDType::ui16:
    return "uint16";
  case KGENDType::si16:
    return "int16";
  case KGENDType::ui32:
    return "uint32";
  case KGENDType::si32:
    return "int32";
  case KGENDType::ui64:
    return "uint64";
  case KGENDType::si64:
    return "int64";
  case KGENDType::ui128:
    return "uint128";
  case KGENDType::si128:
    return "int128";
  case KGENDType::ui256:
    return "uint256";
  case KGENDType::si256:
    return "int256";
#define DECLARE_FLOAT(SHORT_NAME, LONG_NAME, ...)                              \
  case KGENDType::SHORT_NAME:                                                  \
    return #LONG_NAME;
#include "Support/ML/FloatTypes.def"
#undef DECLARE_FLOAT
  default:
    return DType(dtype).getAsString();
  }
}

std::string KGENDType::getAsString(bool libForm) const {
  switch (uint8_t dtype = getValue()) {
  case ExtraCases::index:
    return libForm ? "int" : "index";
  case ExtraCases::uindex:
    return libForm ? "uint" : "uindex";
  case ExtraCases::address:
    return "address";
  default:
    return libForm ? getAsLongString(dtype) : DType::getAsString();
  }
}

std::pair<KGENDType, std::optional<int64_t>>
KGENDType::getEquivalentDType(Type type) {
  auto [dt, vecSize] = M::getEquivalentDType(type);

  KGENDType dtype = KGENDType(dt);
  // `M::getEquivalentDType` does not know the extra case in KGENDType.
  if (dt.isInvalid() && type.isIndex())
    dtype = KGENDType::index;

  return {dtype, vecSize};
}

Type KGENDType::getEquivalentBuiltinType(MLIRContext *ctx) {
  // Bool can only be `i1`.
  if (isBool())
    return IntegerType::get(ctx, 1);

  if (isIndex() || isUIndex())
    return IndexType::get(ctx);

  if (isInt())
    return IntegerType::get(ctx, getWidthInBits(),
                            isSInt() ? IntegerType::Signed
                                     : IntegerType::Unsigned);

  if (isFloat())
    return getEquivalentFloatType(ctx, *this);

  return {};
}
