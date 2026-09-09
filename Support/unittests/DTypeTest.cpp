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

#include "Support/ML/DType.h"
#include "llvm/ADT/StringRef.h"

#include "gtest/gtest.h"

using namespace M;

static std::vector<DType> getAllKnownDTypes() {
  return {
      DType(DType::si1),
      DType(DType::ui1),
      DType(DType::si2),
      DType(DType::ui2),
      DType(DType::si4),
      DType(DType::ui4),
      DType(DType::si8),
      DType(DType::ui8),
      DType(DType::si16),
      DType(DType::ui16),
      DType(DType::si32),
      DType(DType::ui32),
      DType(DType::si64),
      DType(DType::ui64),
      DType(DType::si128),
      DType(DType::ui128),
      DType(DType::f4e2m1fn),
      DType(DType::f6e2m3fn),
      DType(DType::f6e3m2fn),
      DType(DType::f8e8m0fnu),
      DType(DType::f8e5m2),
      DType(DType::f8e5m2fnuz),
      DType(DType::f8e4m3fn),
      DType(DType::f8e4m3fnuz),
      DType(DType::f8e3m4),
      DType(DType::f16),
      DType(DType::f32),
      DType(DType::f64),
      DType(DType::bf16),
      DType(DType::kBool),
      DType::getComplex(DType::si8),
      DType::getComplex(DType::ui8),
      DType::getComplex(DType::si16),
      DType::getComplex(DType::ui16),
      DType::getComplex(DType::si32),
      DType::getComplex(DType::ui32),
      DType::getComplex(DType::si64),
      DType::getComplex(DType::ui64),
      DType::getComplex(DType::si128),
      DType::getComplex(DType::ui128),
      DType::getComplex(DType::f8e5m2),
      DType::getComplex(DType::f8e5m2fnuz),
      DType::getComplex(DType::f8e4m3fn),
      DType::getComplex(DType::f8e4m3fnuz),
      DType::getComplex(DType::f8e3m4),
      DType::getComplex(DType::f16),
      DType::getComplex(DType::f32),
      DType::getComplex(DType::f64),
      DType::getComplex(DType::bf16),
      DType::getComplex(DType::kBool),
  };
}

TEST(DType, getWidthInBits) {
  EXPECT_EQ(-1, DType(DType::invalid).getWidthInBits());
  EXPECT_EQ(1, DType(DType::si1).getWidthInBits());
  EXPECT_EQ(1, DType(DType::ui1).getWidthInBits());
  EXPECT_EQ(2, DType(DType::si2).getWidthInBits());
  EXPECT_EQ(2, DType(DType::ui2).getWidthInBits());
  EXPECT_EQ(4, DType(DType::si4).getWidthInBits());
  EXPECT_EQ(4, DType(DType::ui4).getWidthInBits());
  EXPECT_EQ(8, DType(DType::si8).getWidthInBits());
  EXPECT_EQ(8, DType(DType::ui8).getWidthInBits());
  EXPECT_EQ(16, DType(DType::si16).getWidthInBits());
  EXPECT_EQ(16, DType(DType::ui16).getWidthInBits());
  EXPECT_EQ(32, DType(DType::si32).getWidthInBits());
  EXPECT_EQ(32, DType(DType::ui32).getWidthInBits());
  EXPECT_EQ(64, DType(DType::si64).getWidthInBits());
  EXPECT_EQ(64, DType(DType::ui64).getWidthInBits());
  EXPECT_EQ(128, DType(DType::si128).getWidthInBits());
  EXPECT_EQ(128, DType(DType::ui128).getWidthInBits());
  EXPECT_EQ(4, DType(DType::f4e2m1fn).getWidthInBits());
  EXPECT_EQ(6, DType(DType::f6e2m3fn).getWidthInBits());
  EXPECT_EQ(6, DType(DType::f6e3m2fn).getWidthInBits());
  EXPECT_EQ(8, DType(DType::f8e8m0fnu).getWidthInBits());
  EXPECT_EQ(8, DType(DType::f8e5m2).getWidthInBits());
  EXPECT_EQ(8, DType(DType::f8e5m2fnuz).getWidthInBits());
  EXPECT_EQ(8, DType(DType::f8e4m3fn).getWidthInBits());
  EXPECT_EQ(8, DType(DType::f8e4m3fnuz).getWidthInBits());
  EXPECT_EQ(8, DType(DType::f8e3m4).getWidthInBits());
  EXPECT_EQ(16, DType(DType::f16).getWidthInBits());
  EXPECT_EQ(32, DType(DType::f32).getWidthInBits());
  EXPECT_EQ(64, DType(DType::f64).getWidthInBits());
  EXPECT_EQ(16, DType(DType::bf16).getWidthInBits());
  EXPECT_EQ(8, DType(DType::kBool).getWidthInBits());
  EXPECT_EQ(-1, DType(DType::mIsComplex).getWidthInBits());
  EXPECT_EQ(16, DType::getComplex(DType::si8).getWidthInBits());
  EXPECT_EQ(16, DType::getComplex(DType::ui8).getWidthInBits());
  EXPECT_EQ(32, DType::getComplex(DType::si16).getWidthInBits());
  EXPECT_EQ(32, DType::getComplex(DType::ui16).getWidthInBits());
  EXPECT_EQ(64, DType::getComplex(DType::si32).getWidthInBits());
  EXPECT_EQ(64, DType::getComplex(DType::ui32).getWidthInBits());
  EXPECT_EQ(128, DType::getComplex(DType::si64).getWidthInBits());
  EXPECT_EQ(128, DType::getComplex(DType::ui64).getWidthInBits());
  EXPECT_EQ(256, DType::getComplex(DType::si128).getWidthInBits());
  EXPECT_EQ(256, DType::getComplex(DType::ui128).getWidthInBits());
  EXPECT_EQ(16, DType::getComplex(DType::f8e5m2).getWidthInBits());
  EXPECT_EQ(16, DType::getComplex(DType::f8e4m3fn).getWidthInBits());
  EXPECT_EQ(16, DType::getComplex(DType::f8e3m4).getWidthInBits());
  EXPECT_EQ(32, DType::getComplex(DType::f16).getWidthInBits());
  EXPECT_EQ(64, DType::getComplex(DType::f32).getWidthInBits());
  EXPECT_EQ(128, DType::getComplex(DType::f64).getWidthInBits());
  EXPECT_EQ(32, DType::getComplex(DType::bf16).getWidthInBits());
  EXPECT_EQ(16, DType::getComplex(DType::kBool).getWidthInBits());

  // Verify these aliases line up with getComplex.
  EXPECT_EQ(DType(DType::complex_f32).getValue(),
            DType::getComplex(DType::f32).getValue());
  EXPECT_EQ(DType(DType::complex_f64).getValue(),
            DType::getComplex(DType::f64).getValue());
}

TEST(DType, getSizeInBytesPacked) {
  EXPECT_EQ(DType(DType::si4).getSizeInBytes(2), 1);
  EXPECT_EQ(DType(DType::si4).getSizeInBytes(3), 2);
  EXPECT_EQ(DType(DType::si2).getSizeInBytes(5), 2);
  EXPECT_EQ(DType(DType::si2).getSizeInBytes(4), 1);
  EXPECT_EQ(DType(DType::si1).getSizeInBytes(3), 1);
  EXPECT_EQ(DType(DType::si1).getSizeInBytes(9), 2);
  EXPECT_EQ(DType(DType::f4e2m1fn).getSizeInBytes(1), 1);
  EXPECT_EQ(DType(DType::f4e2m1fn).getSizeInBytes(2), 1);
  EXPECT_EQ(DType(DType::f4e2m1fn).getSizeInBytes(3), 2);
  // FP6 packs 4 elements into exactly 3 bytes.
  EXPECT_EQ(DType(DType::f6e2m3fn).getSizeInBytes(1), 1);
  EXPECT_EQ(DType(DType::f6e2m3fn).getSizeInBytes(2), 2);
  EXPECT_EQ(DType(DType::f6e2m3fn).getSizeInBytes(4), 3);
  EXPECT_EQ(DType(DType::f6e2m3fn).getSizeInBytes(32), 24);
  EXPECT_EQ(DType(DType::f6e3m2fn).getSizeInBytes(4), 3);
}

TEST(DType, getSizeInBytes) {
  EXPECT_EQ(-1, DType(DType::invalid).getSizeInBytes(1));
  for (DType dt : getAllKnownDTypes()) {
    EXPECT_GT(dt.getSizeInBytes(1), 0) << dt.getAsString();

    if (dt.getWidthInBits() % 8 == 0)
      EXPECT_EQ(dt.getSizeInBytes(8), dt.getWidthInBits()) << dt.getAsString();
  }
}

TEST(DType, getSizeInBytesOverflow) {
  EXPECT_GT(DType(DType::f4e2m1fn)
                .getSizeInBytes(std::numeric_limits<ssize_t>::max() >> 1),
            0);
  EXPECT_EQ(-1, DType(DType::f4e2m1fn)
                    .getSizeInBytes(std::numeric_limits<size_t>::max()));
  EXPECT_EQ(-1, DType::getComplex(DType::ui8)
                    .getSizeInBytes(std::numeric_limits<ssize_t>::max()));
}

TEST(DType, getAsString) {
  EXPECT_EQ("invalid", DType(DType::invalid).getAsString());
  EXPECT_EQ("si1", DType(DType::si1).getAsString());
  EXPECT_EQ("ui1", DType(DType::ui1).getAsString());
  EXPECT_EQ("si2", DType(DType::si2).getAsString());
  EXPECT_EQ("ui2", DType(DType::ui2).getAsString());
  EXPECT_EQ("si4", DType(DType::si4).getAsString());
  EXPECT_EQ("ui4", DType(DType::ui4).getAsString());
  EXPECT_EQ("si8", DType(DType::si8).getAsString());
  EXPECT_EQ("ui8", DType(DType::ui8).getAsString());
  EXPECT_EQ("si16", DType(DType::si16).getAsString());
  EXPECT_EQ("ui16", DType(DType::ui16).getAsString());
  EXPECT_EQ("si32", DType(DType::si32).getAsString());
  EXPECT_EQ("ui32", DType(DType::ui32).getAsString());
  EXPECT_EQ("si64", DType(DType::si64).getAsString());
  EXPECT_EQ("ui64", DType(DType::ui64).getAsString());
  EXPECT_EQ("si128", DType(DType::si128).getAsString());
  EXPECT_EQ("ui128", DType(DType::ui128).getAsString());
  EXPECT_EQ("f4e2m1fn", DType(DType::f4e2m1fn).getAsString());
  EXPECT_EQ("f8e8m0fnu", DType(DType::f8e8m0fnu).getAsString());
  EXPECT_EQ("f8e5m2", DType(DType::f8e5m2).getAsString());
  EXPECT_EQ("f8e4m3fnuz", DType(DType::f8e4m3fnuz).getAsString());
  EXPECT_EQ("f8e3m4", DType(DType::f8e3m4).getAsString());
  EXPECT_EQ("f16", DType(DType::f16).getAsString());
  EXPECT_EQ("f32", DType(DType::f32).getAsString());
  EXPECT_EQ("f64", DType(DType::f64).getAsString());
  EXPECT_EQ("bf16", DType(DType::bf16).getAsString());
  EXPECT_EQ("bool", DType(DType::kBool).getAsString());
  EXPECT_EQ("complex<si8>", DType::getComplex(DType::si8).getAsString());
  EXPECT_EQ("complex<ui8>", DType::getComplex(DType::ui8).getAsString());
  EXPECT_EQ("complex<si16>", DType::getComplex(DType::si16).getAsString());
  EXPECT_EQ("complex<ui16>", DType::getComplex(DType::ui16).getAsString());
  EXPECT_EQ("complex<si32>", DType::getComplex(DType::si32).getAsString());
  EXPECT_EQ("complex<ui32>", DType::getComplex(DType::ui32).getAsString());
  EXPECT_EQ("complex<si64>", DType::getComplex(DType::si64).getAsString());
  EXPECT_EQ("complex<ui64>", DType::getComplex(DType::ui64).getAsString());
  EXPECT_EQ("complex<si128>", DType::getComplex(DType::si128).getAsString());
  EXPECT_EQ("complex<ui128>", DType::getComplex(DType::ui128).getAsString());
  EXPECT_EQ("complex<f8e5m2>", DType::getComplex(DType::f8e5m2).getAsString());
  EXPECT_EQ("complex<f8e5m2fnuz>",
            DType::getComplex(DType::f8e5m2fnuz).getAsString());
  EXPECT_EQ("complex<f8e4m3fn>",
            DType::getComplex(DType::f8e4m3fn).getAsString());
  EXPECT_EQ("complex<f8e4m3fnuz>",
            DType::getComplex(DType::f8e4m3fnuz).getAsString());
  EXPECT_EQ("complex<f8e3m4>", DType::getComplex(DType::f8e3m4).getAsString());
  EXPECT_EQ("complex<f16>", DType::getComplex(DType::f16).getAsString());
  EXPECT_EQ("complex<f32>", DType::getComplex(DType::f32).getAsString());
  EXPECT_EQ("complex<f64>", DType::getComplex(DType::f64).getAsString());
  EXPECT_EQ("complex<bf16>", DType::getComplex(DType::bf16).getAsString());
  EXPECT_EQ("complex<bool>", DType::getComplex(DType::kBool).getAsString());
}

TEST(DType, getFromString) {
  EXPECT_EQ(DType(DType::invalid), DType::getFromString("invalid"));
  EXPECT_EQ(DType(DType::si1), DType::getFromString("si1"));
  EXPECT_EQ(DType(DType::ui1), DType::getFromString("ui1"));
  EXPECT_EQ(DType(DType::si2), DType::getFromString("si2"));
  EXPECT_EQ(DType(DType::ui2), DType::getFromString("ui2"));
  EXPECT_EQ(DType(DType::si4), DType::getFromString("si4"));
  EXPECT_EQ(DType(DType::ui4), DType::getFromString("ui4"));
  EXPECT_EQ(DType(DType::si8), DType::getFromString("si8"));
  EXPECT_EQ(DType(DType::ui8), DType::getFromString("ui8"));
  EXPECT_EQ(DType(DType::si16), DType::getFromString("si16"));
  EXPECT_EQ(DType(DType::ui16), DType::getFromString("ui16"));
  EXPECT_EQ(DType(DType::si32), DType::getFromString("si32"));
  EXPECT_EQ(DType(DType::ui32), DType::getFromString("ui32"));
  EXPECT_EQ(DType(DType::si64), DType::getFromString("si64"));
  EXPECT_EQ(DType(DType::ui64), DType::getFromString("ui64"));
  EXPECT_EQ(DType(DType::si128), DType::getFromString("si128"));
  EXPECT_EQ(DType(DType::ui128), DType::getFromString("ui128"));
  EXPECT_EQ(DType(DType::f4e2m1fn), DType::getFromString("f4e2m1fn"));
  EXPECT_EQ(DType(DType::f8e8m0fnu), DType::getFromString("f8e8m0fnu"));
  EXPECT_EQ(DType(DType::f8e5m2), DType::getFromString("f8e5m2"));
  EXPECT_EQ(DType(DType::f8e5m2fnuz), DType::getFromString("f8e5m2fnuz"));
  EXPECT_EQ(DType(DType::f8e4m3fn), DType::getFromString("f8e4m3fn"));
  EXPECT_EQ(DType(DType::f8e4m3fnuz), DType::getFromString("f8e4m3fnuz"));
  EXPECT_EQ(DType(DType::f8e3m4), DType::getFromString("f8e3m4"));
  EXPECT_EQ(DType(DType::f16), DType::getFromString("f16"));
  EXPECT_EQ(DType(DType::f32), DType::getFromString("f32"));
  EXPECT_EQ(DType(DType::f64), DType::getFromString("f64"));
  EXPECT_EQ(DType(DType::bf16), DType::getFromString("bf16"));
  EXPECT_EQ(DType(DType::kBool), DType::getFromString("bool"));
  EXPECT_EQ(DType::getComplex(DType::si8),
            DType::getFromString("complex<si8>"));
  EXPECT_EQ(DType::getComplex(DType::ui8),
            DType::getFromString("complex<ui8>"));
  EXPECT_EQ(DType::getComplex(DType::si16),
            DType::getFromString("complex<si16>"));
  EXPECT_EQ(DType::getComplex(DType::ui16),
            DType::getFromString("complex<ui16>"));
  EXPECT_EQ(DType::getComplex(DType::si32),
            DType::getFromString("complex<si32>"));
  EXPECT_EQ(DType::getComplex(DType::ui32),
            DType::getFromString("complex<ui32>"));
  EXPECT_EQ(DType::getComplex(DType::si64),
            DType::getFromString("complex<si64>"));
  EXPECT_EQ(DType::getComplex(DType::ui64),
            DType::getFromString("complex<ui64>"));
  EXPECT_EQ(DType::getComplex(DType::si128),
            DType::getFromString("complex<si128>"));
  EXPECT_EQ(DType::getComplex(DType::ui128),
            DType::getFromString("complex<ui128>"));
  EXPECT_EQ(DType::getComplex(DType::f8e5m2),
            DType::getFromString("complex<f8e5m2>"));
  EXPECT_EQ(DType::getComplex(DType::f8e5m2fnuz),
            DType::getFromString("complex<f8e5m2fnuz>"));
  EXPECT_EQ(DType::getComplex(DType::f8e4m3fn),
            DType::getFromString("complex<f8e4m3fn>"));
  EXPECT_EQ(DType::getComplex(DType::f8e4m3fnuz),
            DType::getFromString("complex<f8e4m3fnuz>"));
  EXPECT_EQ(DType::getComplex(DType::f8e3m4),
            DType::getFromString("complex<f8e3m4>"));
  EXPECT_EQ(DType::getComplex(DType::f16),
            DType::getFromString("complex<f16>"));
  EXPECT_EQ(DType::getComplex(DType::f32),
            DType::getFromString("complex<f32>"));
  EXPECT_EQ(DType::getComplex(DType::f64),
            DType::getFromString("complex<f64>"));
  EXPECT_EQ(DType::getComplex(DType::bf16),
            DType::getFromString("complex<bf16>"));
  EXPECT_EQ(DType::getComplex(DType::kBool),
            DType::getFromString("complex<bool>"));
}

TEST(DType, CanDispatchOptionals) {
  auto dispatch = [](DType dType) -> std::optional<int> {
    return dType.dispatch<std::optional<int>>()
        .template when<M::DType::si32>([&]() { return 32; })
        .otherwise([&]() { return std::nullopt; });
  };
  EXPECT_EQ(dispatch(DType::si32), 32);
  EXPECT_EQ(dispatch(DType::f32), std::nullopt);
}
