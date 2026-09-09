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
// This file implements Location.h classes.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Support/Location.h"

using namespace M::AsyncRT;

void LocationDecoder::VtableAnchor() {}

/// Decode the location information in this object into a DecodedLocation.
DecodedLocation EncodedLocation::decode() const {
  return decoder->decode(*this);
}
