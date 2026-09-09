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

#ifndef SUPPORT_BASE64_H
#define SUPPORT_BASE64_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include <string>

namespace M {
// TODO: We should contribute better Base64 encoding/decoding to upstream, this
//       should not be necessary.

/// Base-64 encode a string that conforms to RFC4648 Section 5 (URL and filename
/// safe). This implementation does not include the `=` padding at the end of
/// the encoded bytes.
std::string encodeURLSafeBase64(StringRef str);

/// Base-64 decode a string that conforms to RFC4648 Section 5 (URL and filename
/// safe). This implementation does not include the `=` padding at the end of
/// the encoded bytes.
ErrorOr<std::string> decodeURLSafeBase64(StringRef str);
} // namespace M

#endif // SUPPORT_BASE64_H
