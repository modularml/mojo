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

#include "Support/ML/TensorSpec.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "Support/ML/DType.h"
#include "Support/ML/TensorShape.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <utility>

using namespace M;

//===----------------------------------------------------------------------===//
// Parsing methods
//===----------------------------------------------------------------------===//

/// Parses a string of the form dim0xdim1x...xDType into a TensorSpec.
ErrorOr<TensorSpec> TensorSpec::parseFromString(StringRef str) {
  // dtype is the portion after the last 'x' (that precedes "complex", if any).
  // Shape is the portion before that.  If there is no 'x', the whole string is
  // the dtype.  (rsplit almost does this, but the no-'x' case would put the
  // string in shape instead of dtype.)
  auto lastComplexIndex = str.rfind("complex");
  auto lastXIndex = str.rfind(
      'x', lastComplexIndex == StringRef::npos ? str.size() : lastComplexIndex);
  StringRef shapeStr, dtypeStr;
  if (lastXIndex == StringRef::npos) {
    shapeStr = "";
    dtypeStr = str;
  } else {
    shapeStr = str.slice(0, lastXIndex);
    dtypeStr = str.slice(lastXIndex + 1, StringRef::npos);
  }

  auto shape = TensorShape::parseFromString(shapeStr);
  if (failed(shape))
    return Error(Twine("could not parse shape from string: ") + str + ": " +
                 shape.getError());

  auto dtype = DType::getFromString(dtypeStr);
  if (failed(dtype))
    return Error(Twine("could not parse dtype from string: ") + str +
                 " because " + dtypeStr + " is not a valid DType");

  // Create the tensor spec from the shape and dtype information.
  return TensorSpec(*shape, *dtype);
}

//===----------------------------------------------------------------------===//
// Stringification and printing methods
//===----------------------------------------------------------------------===//

void TensorSpec::print(raw_ostream &os) const {
  TensorShape::print(os);
  if (!(hasRank() && getRank() == 0))
    os << "x";
  os << getEltType();
}

std::string TensorSpec::getAsString() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  print(os);
  return os.str();
}

//===----------------------------------------------------------------------===//
// YAML ScalarTraits implementation
//===----------------------------------------------------------------------===//

void llvm::yaml::ScalarTraits<TensorSpec>::output(const M::TensorSpec &value,
                                                  void *ctxt,
                                                  llvm::raw_ostream &out) {
  value.print(out);
}

StringRef llvm::yaml::ScalarTraits<TensorSpec>::input(StringRef scalar,
                                                      void *ctxt,
                                                      M::TensorSpec &value) {
  M::ErrorOr<TensorSpec> specOr = TensorSpec::parseFromString(scalar);
  if (specOr.isError())
    // Can't return specOr.getError() because that has a lifetime coinciding
    // with specOr, whose lifetime ends at the end of this function (can't
    // safely return a StringRef to it, since it would be used after lifetime
    // end).  Unfortunately this means we discard error details, but we don't
    // have the mechanism to preserve them while being safe about lifetime.
    return "Unable to parse tensor spec";
  value = std::move(*specOr);
  return StringRef();
}

llvm::yaml::QuotingType
llvm::yaml::ScalarTraits<TensorSpec>::mustQuote(StringRef) {
  return llvm::yaml::QuotingType::None;
}
