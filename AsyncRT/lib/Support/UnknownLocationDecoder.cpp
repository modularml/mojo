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

#include "AsyncRT/Support/UnknownLocationDecoder.h"
#include "AsyncRT/Support/Diagnostic.h"

using namespace M;
using namespace AsyncRT;

EncodedDiagnostic UnknownLocationDecoder::getDiagnostic(Error err) {
  return {std::move(err), UnknownLocationDecoder::getEncodedLocation()};
}

void UnknownLocationDecoder::addRef() const {
  RCRef<ReferenceCounted<UnknownLocationDecoder>>::lowLevelAddRef(
      const_cast<UnknownLocationDecoder *>(this));
}
void UnknownLocationDecoder::dropRef() const {
  RCRef<ReferenceCounted<UnknownLocationDecoder>>::lowLevelDropRef(
      const_cast<UnknownLocationDecoder *>(this));
}
