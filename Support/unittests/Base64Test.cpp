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

#include "Support/Base64.h"
#include "llvm/ADT/StringRef.h"

#include "gtest/gtest.h"

using namespace M;

/// Just check that a weird string can roundtrip through URL-safe base64.
TEST(Base64, Roundtrip) {
  const llvm::StringLiteral str =
      "This is a string, it has \n and \t in it, maybe some \\ and some \x0A";

  auto encoded = encodeURLSafeBase64(str);
  auto decodedOr = decodeURLSafeBase64(encoded);
  EXPECT_FALSE(decodedOr.isError()) << decodedOr.takeError();
  EXPECT_EQ(str, *decodedOr);
}

/// Check that we add padding correctly. This doesn't actually care about the
/// contents of the string, just that it can be decoded.
TEST(Base64, Padding) {
  // This is a problematic string that originally triggered this bug.
  const llvm::StringLiteral b64Str =
      "SbxQJvZm0NV4rh82C8jWxRMCOhVKm6Oz2UKBcjKCSUA";

  // These are canonical test vectors, but with the padding stripped off.
  const llvm::StringLiteral f = "Zg";
  const llvm::StringLiteral fo = "Zm8";
  const llvm::StringLiteral foo = "Zm9v";
  const llvm::StringLiteral foob = "Zm9vYg";
  const llvm::StringLiteral fooba = "Zm9vYmE";
  const llvm::StringLiteral foobar = "Zm9vYmFy";

  auto decodedOr = decodeURLSafeBase64(b64Str);
  EXPECT_FALSE(decodedOr.isError()) << decodedOr.takeError();

  for (auto str : {f, fo, foo, foob, fooba, foobar}) {
    decodedOr = decodeURLSafeBase64(str);
    EXPECT_FALSE(decodedOr.isError())
        << std::string(str) << ": " << decodedOr.getError();
  }
}
