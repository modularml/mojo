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

#include "Config/include/Config/Version.h"
#include "Mojo/DialectChecksum/DialectChecksum.h"
#include "Mojo/include/Mojo/Support/MojoPrecompiledFile.h"
#include "gtest/gtest.h"

using M::ProjectVersion;
using M::KGEN::MojoPrecompiledFileVersion;

TEST(MojoVersionTest, testNotEmpty) {
  llvm::StringRef checksum = M::getMojoMlirDialectChecksum();
  ASSERT_FALSE(checksum.empty());
}
static MojoPrecompiledFileVersion makeVersion(int maj, int min, int patch,
                                              const char *label) {
  return MojoPrecompiledFileVersion(ProjectVersion{
      maj, min, patch, label, /*revision=*/nullptr, /*buildType=*/nullptr});
}

TEST(MojoVersionTest, testLabelParsing) {
  { // Nothing
    auto mojoVer = makeVersion(1, 8, 9, "");
    EXPECT_EQ(mojoVer.major, 1);
    EXPECT_EQ(mojoVer.minor, 8);
    EXPECT_EQ(mojoVer.patch, 9);
    EXPECT_EQ(mojoVer.abrc, MojoPrecompiledFileVersion::ABRC::None);
    EXPECT_FALSE(mojoVer.nABRC);
    EXPECT_FALSE(mojoVer.postN);
    EXPECT_FALSE(mojoVer.devN);
  }

  { // Full label string - alpha
    auto mojoVer = makeVersion(1, 8, 9, "a0.post3.dev4");
    EXPECT_EQ(mojoVer.major, 1);
    EXPECT_EQ(mojoVer.minor, 8);
    EXPECT_EQ(mojoVer.patch, 9);
    EXPECT_EQ(mojoVer.abrc, MojoPrecompiledFileVersion::ABRC::Alpha);
    EXPECT_EQ(mojoVer.nABRC, 0);
    EXPECT_EQ(mojoVer.postN, 3);
    EXPECT_EQ(mojoVer.devN, 4);
  }

  { // Full label string - beta
    auto mojoVer = makeVersion(1, 8, 9, "b2.post3.dev4");
    EXPECT_EQ(mojoVer.major, 1);
    EXPECT_EQ(mojoVer.minor, 8);
    EXPECT_EQ(mojoVer.patch, 9);
    EXPECT_EQ(mojoVer.abrc, MojoPrecompiledFileVersion::ABRC::Beta);
    EXPECT_EQ(mojoVer.nABRC, 2);
    EXPECT_EQ(mojoVer.postN, 3);
    EXPECT_EQ(mojoVer.devN, 4);
  }

  { // Full label string - RC
    auto mojoVer = makeVersion(1, 8, 9, "rc0.post3.dev4");
    EXPECT_EQ(mojoVer.major, 1);
    EXPECT_EQ(mojoVer.minor, 8);
    EXPECT_EQ(mojoVer.patch, 9);
    EXPECT_EQ(mojoVer.abrc, MojoPrecompiledFileVersion::ABRC::RC);
    EXPECT_EQ(mojoVer.nABRC, 0);
    EXPECT_EQ(mojoVer.postN, 3);
    EXPECT_EQ(mojoVer.devN, 4);
  }

  { // Only {a|b|rc}N
    auto mojoVer = makeVersion(1, 8, 9, "b0");
    EXPECT_EQ(mojoVer.major, 1);
    EXPECT_EQ(mojoVer.minor, 8);
    EXPECT_EQ(mojoVer.patch, 9);
    EXPECT_EQ(mojoVer.abrc, MojoPrecompiledFileVersion::ABRC::Beta);
    EXPECT_EQ(mojoVer.nABRC, 0);
    EXPECT_FALSE(mojoVer.postN);
    EXPECT_FALSE(mojoVer.devN);
  }

  { // Only {a|b|rc}N.postN
    auto mojoVer = makeVersion(1, 8, 9, "b0.post0");
    EXPECT_EQ(mojoVer.major, 1);
    EXPECT_EQ(mojoVer.minor, 8);
    EXPECT_EQ(mojoVer.patch, 9);
    EXPECT_EQ(mojoVer.abrc, MojoPrecompiledFileVersion::ABRC::Beta);
    EXPECT_EQ(mojoVer.nABRC, 0);
    EXPECT_EQ(mojoVer.postN, 0);
    EXPECT_FALSE(mojoVer.devN);
  }

  { // Only {a|b|rc}N.devN
    auto mojoVer = makeVersion(1, 8, 9, "b0.dev4");
    EXPECT_EQ(mojoVer.major, 1);
    EXPECT_EQ(mojoVer.minor, 8);
    EXPECT_EQ(mojoVer.patch, 9);
    EXPECT_EQ(mojoVer.abrc, MojoPrecompiledFileVersion::ABRC::Beta);
    EXPECT_EQ(mojoVer.nABRC, 0);
    EXPECT_FALSE(mojoVer.postN);
    EXPECT_EQ(mojoVer.devN, 4);
  }

  { // Only .postN.devN
    auto mojoVer = makeVersion(1, 8, 9, ".post3.dev4");
    EXPECT_EQ(mojoVer.major, 1);
    EXPECT_EQ(mojoVer.minor, 8);
    EXPECT_EQ(mojoVer.patch, 9);
    EXPECT_EQ(mojoVer.abrc, MojoPrecompiledFileVersion::ABRC::None);
    EXPECT_FALSE(mojoVer.nABRC);
    EXPECT_EQ(mojoVer.postN, 3);
    EXPECT_EQ(mojoVer.devN, 4);
  }

  { // Only .devN
    auto mojoVer = makeVersion(1, 8, 9, ".dev4");
    EXPECT_EQ(mojoVer.major, 1);
    EXPECT_EQ(mojoVer.minor, 8);
    EXPECT_EQ(mojoVer.patch, 9);
    EXPECT_EQ(mojoVer.abrc, MojoPrecompiledFileVersion::ABRC::None);
    EXPECT_FALSE(mojoVer.nABRC);
    EXPECT_FALSE(mojoVer.postN);
    EXPECT_EQ(mojoVer.devN, 4);
  }
}

TEST(MojoVersionTest, testInvalidLabelParsing) {
  {
    auto mojoVer = makeVersion(1, 8, 9, "b-1.post3.dev4");
    EXPECT_EQ(mojoVer.major, 1);
    EXPECT_EQ(mojoVer.minor, 8);
    EXPECT_EQ(mojoVer.patch, 9);
    // The label parsing will have bailed out and reset the label field.
    EXPECT_EQ(mojoVer.abrc, MojoPrecompiledFileVersion::ABRC::None);
    EXPECT_FALSE(mojoVer.nABRC);
    EXPECT_FALSE(mojoVer.postN);
    EXPECT_FALSE(mojoVer.devN);
  }

  {
    auto mojoVer0 = makeVersion(1, 8, 9, "");
    auto mojoVer1 = makeVersion(1, 8, 9, "garbage");
    auto mojoVer2 = makeVersion(1, 8, 9, "b10garbage");
    EXPECT_TRUE(mojoVer0.hasValidLabel);
    EXPECT_FALSE(mojoVer1.hasValidLabel);
    EXPECT_FALSE(mojoVer2.hasValidLabel);
    EXPECT_EQ(mojoVer0, mojoVer1);
    EXPECT_EQ(mojoVer0, mojoVer2);
    EXPECT_EQ(mojoVer1, mojoVer2);
  }
}

TEST(MojoVersionTest, testVersionOrdering) {
  // 0.26.3.devN < 1.0.0b1.devN < 1.0.0b1.devN+1 < 1.0.0b1 < 1.0.0b2.devN
  // < 1.0.0b3 < 1.0.0rcN < 1.0.0 < 1.0.0.postN < 1.0.0.postN+1.devN
  // < 1.0.0.postN+1 < 1.1.0
  auto ver0 = makeVersion(0, 26, 3, ".dev0");
  auto ver1 = makeVersion(1, 0, 0, "a1.dev0");
  auto ver2 = makeVersion(1, 0, 0, "a1.dev1");
  auto ver3 = makeVersion(1, 0, 0, "b1.dev0");
  auto ver4 = makeVersion(1, 0, 0, "b1.dev1");
  auto ver5 = makeVersion(1, 0, 0, "b1");
  auto ver6 = makeVersion(1, 0, 0, "b2.dev0");
  auto ver7 = makeVersion(1, 0, 0, "b3");
  auto ver8 = makeVersion(1, 0, 0, "rc0");
  auto ver9 = makeVersion(1, 0, 0, "");
  auto ver10 = makeVersion(1, 0, 0, ".post0");
  auto ver11 = makeVersion(1, 0, 0, ".post1.dev0");
  auto ver12 = makeVersion(1, 0, 0, ".post1");
  auto ver13 = makeVersion(1, 1, 0, "");

  std::vector<MojoPrecompiledFileVersion> versions = {
      ver0, ver1, ver2, ver3,  ver4,  ver5,  ver6,
      ver7, ver8, ver9, ver10, ver11, ver12, ver13};

  for (unsigned i = 0, e = versions.size() - 1; i != e; i++) {
    EXPECT_NE(versions[i], versions[i + 1]);
    EXPECT_LT(versions[i], versions[i + 1]) << i << " < " << i + 1;
    EXPECT_GT(versions[i + 1], versions[i]) << i + 1 << " > " << i;
  }

  {
    // 1.0.0.dev0 must sort BEFORE 1.0.0a1
    auto final_dev = makeVersion(1, 0, 0, ".dev0");
    auto alpha1 = makeVersion(1, 0, 0, "a1");
    EXPECT_LT(final_dev, alpha1);
  }
}

TEST(MojoVersionTest, testVersionEquality) {
  EXPECT_EQ(makeVersion(1, 0, 0, ".dev"), makeVersion(1, 0, 0, ".dev0"));
  EXPECT_EQ(makeVersion(1, 0, 0, ".post"), makeVersion(1, 0, 0, ".post0"));
  EXPECT_EQ(makeVersion(1, 0, 0, ".post.dev"),
            makeVersion(1, 0, 0, ".post.dev0"));
  EXPECT_EQ(makeVersion(1, 0, 0, ".post.dev"),
            makeVersion(1, 0, 0, ".post0.dev0"));

  EXPECT_EQ(makeVersion(1, 0, 0, "alpha0"), makeVersion(1, 0, 0, "a0"));
  EXPECT_EQ(makeVersion(1, 0, 0, "beta1"), makeVersion(1, 0, 0, "b1"));
  EXPECT_EQ(makeVersion(1, 0, 0, "rc2"), makeVersion(1, 0, 0, "c2"));
}
