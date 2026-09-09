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

#include "Support/DeviceSpecs.h"

#include "gtest/gtest.h"

using namespace M;

namespace {

static DeviceSpecCollection createDeviceSpecCollection() {
  DeviceSpecCollection specs;
  specs.host.label = "cpu";

  {
    DeviceSpec &spec = specs.devices.emplace_back();
    spec.ref.label = "cpu";
    spec.target.triple = llvm::Triple("x86_64-unknown-linux-gnu");
    spec.target.arch = "znver3";
    spec.target.features.emplace_back("avx2");
    spec.target.features.emplace_back("avx");
  }

  {
    DeviceSpec &spec = specs.devices.emplace_back();
    spec.ref.label = "cuda";
    spec.ref.id = 0;
    spec.target.triple = llvm::Triple("nvptx64-nvidia-cuda");
    spec.target.arch = "sm_80";
  }

  {
    DeviceSpec &spec = specs.devices.emplace_back();
    spec.ref.label = "cuda";
    spec.ref.id = 1;
    spec.target.triple = llvm::Triple("nvptx64-nvidia-cuda");
    spec.target.arch = "sm_80";
  }

  return specs;
}

TEST(DevicesSpecs, RoundTrip) {
  DeviceSpecCollection specs = createDeviceSpecCollection();
  ErrorOr<DeviceSpecCollection> roundTrippedOr =
      specs.deserializeFromJSON(specs.serializeToJSON());
  ASSERT_FALSE(roundTrippedOr.isError());
  EXPECT_EQ(roundTrippedOr->serializeToJSON(), specs.serializeToJSON());
}

TEST(DeviceSpecs, DisabledFeaturesRoundTrip) {
  DeviceSpec spec;
  spec.ref.label = "cpu";
  spec.target.triple = llvm::Triple("x86_64-unknown-linux-gnu");
  spec.target.arch = "znver3";
  spec.target.features = {"avx2", "bmi1"};
  spec.target.disabledFeatures = {"avx512f"};

  ErrorOr<DeviceSpec> roundTrippedOr =
      DeviceSpec::deserializeFromJSON(spec.serializeToJSON());
  ASSERT_FALSE(roundTrippedOr.isError());
  EXPECT_EQ(roundTrippedOr->target.features, spec.target.features);
  EXPECT_EQ(roundTrippedOr->target.disabledFeatures,
            spec.target.disabledFeatures);
}

TEST(DeviceSpecs, ABIRoundTrip) {
  DeviceSpec spec;
  spec.ref.label = "cpu";
  spec.target.triple = llvm::Triple("riscv64-unknown-linux-gnu");
  spec.target.arch = "generic-rv64";
  spec.target.abi = "lp64d";

  ErrorOr<DeviceSpec> roundTrippedOr =
      DeviceSpec::deserializeFromJSON(spec.serializeToJSON());
  ASSERT_FALSE(roundTrippedOr.isError());
  EXPECT_EQ(roundTrippedOr->target.abi, spec.target.abi);
}

TEST(DeviceSpecs, CheckSatisfiesRequirementsABI) {
  TargetInfo provided;
  provided.triple = llvm::Triple("riscv64-unknown-linux-gnu");
  provided.arch = "generic-rv64";
  provided.abi = "lp64d";

  // No ABI requirement is always satisfied.
  TargetInfo noABIRequired = provided;
  noABIRequired.abi.clear();
  EXPECT_FALSE(provided.checkSatisfiesRequirements(noABIRequired).isError());

  // A matching ABI requirement is satisfied.
  TargetInfo sameABIRequired = provided;
  EXPECT_FALSE(provided.checkSatisfiesRequirements(sameABIRequired).isError());

  // A different ABI requirement is not satisfied; unlike arch/features, ABI
  // compatibility is exact-or-nothing.
  TargetInfo differentABIRequired = provided;
  differentABIRequired.abi = "lp64";
  EXPECT_TRUE(
      provided.checkSatisfiesRequirements(differentABIRequired).isError());
}

TEST(DeviceSpecs, FindDeviceSpec) {
  DeviceSpecCollection specs = createDeviceSpecCollection();
  EXPECT_EQ(specs.getHostDeviceSpec().ref.toString(), "cpu:0");
  {
    ErrorOr<const DeviceSpec *> specOr =
        specs.findDeviceSpec(DeviceRef("cuda"));
    ASSERT_FALSE(specOr.isError());
    EXPECT_EQ((*specOr)->ref.toString(), "cuda:0");
  }
  {
    ErrorOr<const DeviceSpec *> specOr =
        specs.findDeviceSpec(DeviceRef("cuda", 1));
    ASSERT_FALSE(specOr.isError());
    EXPECT_EQ((*specOr)->ref.toString(), "cuda:1");
  }
  {
    ErrorOr<const DeviceSpec *> specOr =
        specs.findDeviceSpec(DeviceRef("cuda", 2));
    EXPECT_TRUE(specOr.isError());
  }
}

TEST(DeviceSpecs, EncodeFeaturesUnsigned) {
  TargetInfo ti(llvm::Triple(""), "", {"avx2", "bmi1"});
  EXPECT_EQ(encodeFeatures(ti), "+avx2,+bmi1");
}

TEST(DeviceSpecs, EncodeFeaturesWithDisabled) {
  TargetInfo ti(llvm::Triple(""), "", {"avx2", "bmi1"}, {"avx512f"});
  EXPECT_EQ(encodeFeatures(ti), "+avx2,+bmi1,-avx512f");
}

TEST(DeviceSpecs, DecodeFeaturesPositive) {
  ErrorOr<DecodedFeatures> result = decodeFeatures("+avx2,+bmi1");
  ASSERT_FALSE(result.isError());
  EXPECT_EQ(result->enabled, (std::vector<std::string>{"avx2", "bmi1"}));
  EXPECT_TRUE(result->disabled.empty());
}

TEST(DeviceSpecs, DecodeFeaturesNegative) {
  ErrorOr<DecodedFeatures> result =
      decodeFeatures("+avx2,+bmi1,-avx512f,-avx512bw");
  ASSERT_FALSE(result.isError());
  EXPECT_EQ(result->enabled, (std::vector<std::string>{"avx2", "bmi1"}));
  EXPECT_EQ(result->disabled,
            (std::vector<std::string>{"avx512f", "avx512bw"}));
}

TEST(DeviceSpecs, EncodeDecodeRoundTrip) {
  TargetInfo ti(llvm::Triple(""), "", {"avx2", "bmi1"}, {"avx512f"});
  std::string encoded = encodeFeatures(ti);
  EXPECT_EQ(encoded, "+avx2,+bmi1,-avx512f");
  ErrorOr<DecodedFeatures> decoded = decodeFeatures(encoded);
  ASSERT_FALSE(decoded.isError());
  EXPECT_EQ(decoded->enabled, ti.features);
  EXPECT_EQ(decoded->disabled, ti.disabledFeatures);
}

TEST(DeviceSpecs, HasFeature) {
  // Basic enabled/disabled checks.
  EXPECT_TRUE(hasFeature("+avx2,+avx512f,-long-calls", "avx2"));
  EXPECT_TRUE(hasFeature("+avx2,+avx512f,-long-calls", "avx512f"));
  EXPECT_FALSE(hasFeature("+avx2,+avx512f,-long-calls", "long-calls"));
  EXPECT_FALSE(hasFeature("+avx2,+avx512f,-long-calls", "neon"));

  // No accidental prefix/substring matching: "avx512" must not match
  // "+avx512f".
  EXPECT_FALSE(hasFeature("+avx512f,+avx2", "avx512"));

  // Disabled token must not match as enabled.
  EXPECT_FALSE(hasFeature("+avx2,-avx512f", "avx512f"));
}

TEST(DeviceSpecs, SimdWidthFromFeature) {
  EXPECT_EQ(simdWidthFromFeature("avx512f"), 512u);
  EXPECT_EQ(simdWidthFromFeature("avx2"), 256u);
  EXPECT_EQ(simdWidthFromFeature("bmi1"), 128u);

  // The answer is in bits whatever unit the feature spells its length in: HVX
  // names its vector length in bytes, so "length128b" is a 1024-bit register.
  EXPECT_EQ(simdWidthFromFeature("hvx-length128b"), 1024u);
  EXPECT_EQ(simdWidthFromFeature("hvx-length64b"), 512u);
  // Only the length feature carries a width; the rest of the HVX set does not.
  EXPECT_EQ(simdWidthFromFeature("hvx"), 128u);
  EXPECT_EQ(simdWidthFromFeature("hvxv81"), 128u);
}

// The whole-string form takes the widest enabled feature. The x86 and AArch64
// rows pin that recognizing HVX did not move any other target, and the last row
// pins that a disabled token cannot raise the width.
TEST(DeviceSpecs, SimdWidthFromFeatureString) {
  EXPECT_EQ(simdWidthFromFeatures("+sse4.2"), 128u);
  EXPECT_EQ(simdWidthFromFeatures("+neon,+dotprod,+i8mm"), 128u);
  EXPECT_EQ(simdWidthFromFeatures("+avx,+avx2,+bmi1"), 256u);
  EXPECT_EQ(simdWidthFromFeatures("+avx512f,+avx2,+avx"), 512u);
  EXPECT_EQ(simdWidthFromFeatures(
                "+hvx,+hvx-length128b,+hvx-qfloat,+hvxv81,+v81,-long-calls"),
            1024u);
  EXPECT_EQ(simdWidthFromFeatures("+hvx,+hvx-ieee-fp,+hvx-length64b,+hvxv68"),
            512u);
  EXPECT_EQ(simdWidthFromFeatures("+avx2,-hvx-length128b"), 256u);
}

TEST(DeviceSpecs, DecodeFeaturesLastWins) {
  // Last entry wins: +avx512f,-avx512f resolves to disabled.
  {
    ErrorOr<DecodedFeatures> result = decodeFeatures("+avx512f,-avx512f");
    ASSERT_FALSE(result.isError());
    EXPECT_TRUE(result->enabled.empty());
    EXPECT_EQ(result->disabled, (std::vector<std::string>{"avx512f"}));
  }
  {
    ErrorOr<DecodedFeatures> result = decodeFeatures("-avx512f,+avx512f");
    ASSERT_FALSE(result.isError());
    EXPECT_EQ(result->enabled, (std::vector<std::string>{"avx512f"}));
    EXPECT_TRUE(result->disabled.empty());
  }
}

TEST(DeviceSpecs, DecodeFeaturesNormalizesUnsigned) {
  // Unsigned names are treated as enabled for backward compat with older
  // serialized TargetInfos that predate the signed format.
  ErrorOr<DecodedFeatures> result = decodeFeatures("avx2,bmi1");
  ASSERT_FALSE(result.isError());
  EXPECT_EQ(result->enabled, (std::vector<std::string>{"avx2", "bmi1"}));
  EXPECT_TRUE(result->disabled.empty());

  result = decodeFeatures("+avx2,bmi1");
  ASSERT_FALSE(result.isError());
  EXPECT_EQ(result->enabled, (std::vector<std::string>{"avx2", "bmi1"}));
}

} // namespace
