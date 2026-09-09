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

#include "Support/URI.h"

#include "gtest/gtest.h"

using namespace M;

TEST(URITest, filesystem) {
  // When constructing an URI from a std::filesystem::path, the URI
  // must preserve and not modify the path.
  std::filesystem::path relativePath = "this/is/a/relative/path";
  std::filesystem::path absolutePath = "/this/is/an/absolute/path";

  URI uriRel(relativePath);
  EXPECT_EQ(uriRel.getScheme(), "file");
  EXPECT_TRUE(uriRel.getAuthority().empty());
  EXPECT_EQ(std::filesystem::path(uriRel.getPath().str()), relativePath);

  URI uriAbs(absolutePath);
  EXPECT_EQ(uriAbs.getScheme(), "file");
  EXPECT_TRUE(uriAbs.getAuthority().empty());
  EXPECT_EQ(std::filesystem::path(uriAbs.getPath().str()), absolutePath);
}

TEST(URITest, parseWindowsPath) {
  ErrorOr<URI> uriOr = URI::parse("c:\foo\bar");
  EXPECT_FALSE(uriOr.isError());
  EXPECT_EQ((*uriOr).getScheme(), "file");
  EXPECT_TRUE((*uriOr).getAuthority().empty());
  EXPECT_EQ((*uriOr).getPath(), "c:\foo\bar");

  uriOr = URI::parse("Z:/foo/bar");
  EXPECT_FALSE(uriOr.isError());
  EXPECT_EQ((*uriOr).getScheme(), "file");
  EXPECT_TRUE((*uriOr).getAuthority().empty());
  EXPECT_EQ((*uriOr).getPath(), "Z:/foo/bar");
}

TEST(URITest, parseS3) {
  ErrorOr<URI> uriOr = URI::parse("s3://bucketname/a/path");
  EXPECT_FALSE(uriOr.isError());
  EXPECT_EQ((*uriOr).getScheme(), "s3");
  EXPECT_EQ((*uriOr).getAuthority(), "bucketname");
  EXPECT_EQ((*uriOr).getPath(), "/a/path");
}

TEST(URITest, parseHttp) {
  ErrorOr<URI> uriOr = URI::parse("http://github.com/modularml/modular");
  EXPECT_FALSE(uriOr.isError());
  EXPECT_EQ((*uriOr).getScheme(), "http");
  EXPECT_EQ((*uriOr).getAuthority(), "github.com");
  EXPECT_EQ((*uriOr).getPath(), "/modularml/modular");
}

TEST(URITest, parseError) {
  ErrorOr<URI> uriOr = URI::parse("0123://github.com/some/path");
  EXPECT_TRUE(uriOr.isError());
  EXPECT_STREQ("Invalid scheme: 0123", uriOr.getError());
}
