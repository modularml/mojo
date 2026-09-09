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

#include "Support/MArchTarget/MArchTarget.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/MDialect/MDialect.h"
#include "Support/PlatformUtils.h"
#include "llvm/Support/TargetSelect.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace M;

TEST(ArchTarget, GetFeatures) {
  // Initialize the LLVM targets so we can look up the current target machine.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  MLIRContext ctx{MLIRContext::Threading::DISABLED};
  ctx.loadDialect<MDialect>();
  auto targetInfo =
      M::getMArchFeatures(&ctx, "x86_64-unknown-linux-gnu", "skylake-avx512",
                          "generic", "", "", llvm::Reloc::Static);
  ASSERT_FALSE(targetInfo.isError()) << targetInfo.getError();
  EXPECT_EQ(targetInfo->getRelocationModel(), llvm::Reloc::Static);

  targetInfo = M::getMArchFeatures(&ctx, "x86_64-apple-macosx11.0", "x86-64",
                                   "apple", "", "", llvm::Reloc::PIC_);
  ASSERT_FALSE(targetInfo.isError()) << targetInfo.getError();
  EXPECT_EQ(targetInfo->getArch(), "x86-64");
  EXPECT_EQ(targetInfo->getRelocationModel(), llvm::Reloc::PIC_);

  targetInfo = M::getMArchFeatures(&ctx, "arm64-apple-macosx11.0", "arm64",
                                   "apple-m1", "", "", llvm::Reloc::PIC_);
  ASSERT_FALSE(targetInfo.isError()) << targetInfo.getError();
  EXPECT_EQ(targetInfo->getArch(), "apple-m1");
}

TEST(ArchTarget, getMArchTargetInfo) {
  llvm::InitializeAllTargets();

  ErrorOr<TargetInfo> info = M::getMArchTargetInfo(
      "aarch64-unknown-linux-gnu", "armv8.2-a", "neoverse-n1", "");
  ASSERT_FALSE(info.isError()) << info.getError();
  EXPECT_EQ(info->arch, "neoverse-n1");
  EXPECT_EQ(info->triple.str(), "aarch64-unknown-linux-gnu");
}

// A host CPU LLVM cannot identify must not take the whole target down with it.
// getHostCPUName() answers "generic" for such a part, and Clang rejects that
// name on x86_64 -- so a part newer than the pinned LLVM's CPU tables otherwise
// fails outright. Driving the fallback directly covers that without the
// hardware.
TEST(ArchTarget, UnrecognizedCpuFallsBackToBaseline) {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  MLIRContext ctx{MLIRContext::Threading::DISABLED};
  ctx.loadDialect<MDialect>();

  // Each pair is a (triple, CPU name) that Clang rejects: the literal
  // getHostCPUName() answer for an unidentified x86_64 part, and a name that
  // can never become valid, which keeps the AArch64 side covered no matter how
  // LLVM's CPU tables move.
  const std::pair<StringRef, StringRef> rejected[] = {
      {"x86_64-unknown-linux-gnu", "generic"},
      {"aarch64-unknown-linux-gnu", "not-a-real-cpu"},
  };

  for (auto [triple, name] : rejected) {
    // The name really is one the target refuses, so the fallback below is doing
    // work rather than passing a valid name through.
    ASSERT_TRUE(M::getFeatures(triple, name).isError())
        << triple << " unexpectedly accepts '" << name << "'";

    ErrorOr<M::ResolvedCpu> resolvedOr = M::resolveCpu(triple, name);
    ASSERT_FALSE(resolvedOr.isError())
        << triple << ": " << resolvedOr.getError();
    M::ResolvedCpu resolved = resolvedOr.takeValue();
    EXPECT_TRUE(resolved.name.empty()) << triple << " kept the rejected name";

    // The baseline the fallback picked must still yield a usable target.
    ErrorOr<TargetInfoAttr> targetOr = M::getTargetInfoFor(
        &ctx, triple, resolved.name,
        encodeFeatures(TargetInfo({}, {}, std::move(resolved.features))),
        /*tuneCpu=*/"", /*acceleratorArch=*/"", llvm::Reloc::Static);
    ASSERT_FALSE(targetOr.isError()) << triple << ": " << targetOr.getError();
  }
}

// The fallback must fire only for a name the target rejects; a CPU LLVM knows
// keeps its name, and with it the model-specific features.
TEST(ArchTarget, RecognizedCpuKeepsItsName) {
  ErrorOr<M::ResolvedCpu> resolvedOr =
      M::resolveCpu("aarch64-unknown-linux-gnu", "neoverse-n1");
  ASSERT_FALSE(resolvedOr.isError()) << resolvedOr.getError();
  EXPECT_EQ(resolvedOr->name, "neoverse-n1");
}

// The host must always describe itself, whatever LLVM makes of its CPU, and the
// name it reports must be one the target accepts -- the features are resolved
// for that name, and it is handed on to whoever builds the target machine. On a
// host LLVM cannot identify this is what fails without the fallback.
TEST(ArchTarget, HostTargetInfoIsAlwaysConstructible) {
  ErrorOr<TargetInfo> infoOr = M::getHostTargetInfo();
  ASSERT_FALSE(infoOr.isError()) << infoOr.getError();
  EXPECT_FALSE(M::getFeatures(infoOr->triple.str(), infoOr->arch).isError())
      << "host reports CPU '" << infoOr->arch
      << "' that its own target rejects";
}

// getTargetInfoFor must expand the explicit --target-features delta against
// the CPU model defaults so that hasFeature() reflects what LLVM will actually
// compile for. znver4 enables avx512f by default; omitting -avx512f from the
// feature string should not make it invisible to Mojo's compile-time queries.
TEST(ArchTarget, GetTargetInfoForExpandsFeatures) {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  MLIRContext ctx{MLIRContext::Threading::DISABLED};
  ctx.loadDialect<MDialect>();

  constexpr StringLiteral triple = "x86_64-unknown-linux-gnu";
  constexpr StringLiteral cpu = "znver4";
  // Feature string that enables/disables some other features but does not
  // mention avx512f
  constexpr StringLiteral featuresWithoutDisablingAvx512f =
      "+avx,+avx2,-avx512bw,-avx512cd,-avx512dq,-avx512vl";

  // avx512f is part of znver4's CPU model defaults. Without an explicit
  // -avx512f in the feature string, LLVM keeps it enabled — hasFeature must
  // agree.
  auto targetOn =
      M::getTargetInfoFor(&ctx, triple, cpu, featuresWithoutDisablingAvx512f,
                          "", "", llvm::Reloc::Static);
  ASSERT_FALSE(targetOn.isError()) << targetOn.getError();
  EXPECT_TRUE(targetOn->hasFeature("avx512f"));
  EXPECT_TRUE(targetOn->hasFeature("avx2"));

  // With an explicit -avx512f, hasFeature must return false.
  constexpr StringLiteral featuresWithAvx512fDisabled =
      "+avx,+avx2,-avx512bw,-avx512cd,-avx512dq,-avx512f,-avx512vl";
  auto targetOff =
      M::getTargetInfoFor(&ctx, triple, cpu, featuresWithAvx512fDisabled, "",
                          "", llvm::Reloc::Static);
  ASSERT_FALSE(targetOff.isError()) << targetOff.getError();
  EXPECT_FALSE(targetOff->hasFeature("avx512f"));
  EXPECT_TRUE(targetOff->hasFeature("avx2"));
}

// Both RISC-V spellings must reach hasFeature(), including the extensions each
// one implies, or `CompilationTarget.has_riscv_extension()` is blind.
TEST(ArchTarget, RISCVExpandsExtensions) {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  MLIRContext ctx{MLIRContext::Threading::DISABLED};
  ctx.loadDialect<MDialect>();

  // sifive-e31 is RV32IMAC; the CPU name alone must yield its extensions.
  auto fromCpu = M::getMArchFeatures(&ctx, "riscv32-unknown-none-elf", "",
                                     "sifive-e31", "", "", llvm::Reloc::Static);
  ASSERT_FALSE(fromCpu.isError()) << fromCpu.getError();
  EXPECT_EQ(fromCpu->getTriple().str(), "riscv32-unknown-none-elf");
  EXPECT_TRUE(fromCpu->hasFeature("i"));
  EXPECT_TRUE(fromCpu->hasFeature("m"));
  EXPECT_TRUE(fromCpu->hasFeature("a"));
  EXPECT_TRUE(fromCpu->hasFeature("c"));
  // Implied by M and A respectively.
  EXPECT_TRUE(fromCpu->hasFeature("zmmul"));
  EXPECT_TRUE(fromCpu->hasFeature("zaamo"));
  EXPECT_FALSE(fromCpu->hasFeature("d"));

  // An -march ISA string is the idiomatic RISC-V spelling.
  auto fromMarch =
      M::getMArchFeatures(&ctx, "riscv32-unknown-none-elf", "rv32i_zba_zbb", "",
                          "", "", llvm::Reloc::Static);
  ASSERT_FALSE(fromMarch.isError()) << fromMarch.getError();
  EXPECT_EQ(fromMarch->getTriple().str(), "riscv32-unknown-none-elf");
  EXPECT_TRUE(fromMarch->hasFeature("zba"));
  EXPECT_TRUE(fromMarch->hasFeature("zbb"));
  EXPECT_FALSE(fromMarch->hasFeature("m"));

  // A bare RV64 CPU still reports the base ISA and nothing it lacks.
  auto rv64 = M::getMArchFeatures(&ctx, "riscv64-unknown-none-elf", "",
                                  "generic-rv64", "", "", llvm::Reloc::Static);
  ASSERT_FALSE(rv64.isError()) << rv64.getError();
  EXPECT_TRUE(rv64->hasFeature("i"));
  EXPECT_FALSE(rv64->hasFeature("m"));
}

// stdlib_plugin is a string field on TargetInfoAttr, defaulting to
// "default".
TEST(ArchTarget, StdlibPlugin) {
  MLIRContext ctx{MLIRContext::Threading::DISABLED};
  ctx.loadDialect<MDialect>();

  // Explicitly passing "default" matches the field default.
  auto defaultAttr = TargetInfoAttr::get(
      &ctx, llvm::Triple("a"), "b", /*stdlib_plugin=*/"default",
      /*features=*/"", /*data_layout=*/{},
      /*relocation_model=*/llvm::Reloc::Static,
      /*simd_bit_width=*/0, /*index_bit_width=*/std::nullopt,
      /*tune_cpu=*/"", /*accelerator_arch=*/"", /*abi=*/"");
  EXPECT_EQ(defaultAttr.getStdlibPlugin(), "default");

  // A non-default plugin name is stored on and read back from the attribute.
  auto pluginAttr = TargetInfoAttr::get(
      &ctx, llvm::Triple("a"), "b", /*stdlib_plugin=*/"metal",
      /*features=*/"", /*data_layout=*/{},
      /*relocation_model=*/llvm::Reloc::Static,
      /*simd_bit_width=*/0, /*index_bit_width=*/std::nullopt,
      /*tune_cpu=*/"", /*accelerator_arch=*/"", /*abi=*/"");
  EXPECT_EQ(pluginAttr.getStdlibPlugin(), "metal");
}

// DataLayout parses pointer specs per address space. The spec format is
// `p[<addr-space>]:<size>:<abi-align>[:...]` (LLVM-compatible); `p` with no
// number is the default address space (0). Each address space keeps its own
// pointer width and ABI alignment, so a target with, e.g., 32-bit GPU shared
// pointers and 64-bit global pointers round-trips correctly.
TEST(ArchTarget, DataLayoutPointerAddressSpaces) {
  // Default address space only: `p:64:64` -> 64-bit width, 8-byte ABI align.
  // The no-arg and explicit-zero queries are equivalent.
  {
    auto dl = M::DataLayout::parse("e-p:64:64-i64:64");
    ASSERT_FALSE(dl.isError()) << dl.getError();
    EXPECT_EQ(dl->getPointerBitWidth(), 64);
    EXPECT_EQ(dl->getPointerBitWidth(/*addrSpace=*/0), 64);
    EXPECT_EQ(dl->getPointerABIAlign(), 8);
  }

  // A non-default pointer size for the default address space parses.
  {
    auto dl = M::DataLayout::parse("e-p:32:32-i64:64");
    ASSERT_FALSE(dl.isError()) << dl.getError();
    EXPECT_EQ(dl->getPointerBitWidth(), 32);
    EXPECT_EQ(dl->getPointerABIAlign(), 4);
  }

  // Multiple address spaces with distinct widths/alignments are each parsed and
  // queryable independently (only the explicitly specified spaces are queried).
  {
    auto dl = M::DataLayout::parse("e-p:64:64-p3:32:32-p5:128:128-i64:64");
    ASSERT_FALSE(dl.isError()) << dl.getError();
    // Default (address space 0): 64-bit, 8-byte aligned.
    EXPECT_EQ(dl->getPointerBitWidth(0), 64);
    EXPECT_EQ(dl->getPointerABIAlign(0), 8);
    // Address space 3 (e.g. GPU shared/LDS): 32-bit, 4-byte aligned.
    EXPECT_EQ(dl->getPointerBitWidth(3), 32);
    EXPECT_EQ(dl->getPointerABIAlign(3), 4);
    // Address space 5: 128-bit, 16-byte aligned.
    EXPECT_EQ(dl->getPointerBitWidth(5), 128);
    EXPECT_EQ(dl->getPointerABIAlign(5), 16);
  }

  // The trailing index-size field (`p7:128:128:128:32`) is accepted and ignored
  // for sizing/alignment purposes.
  {
    auto dl = M::DataLayout::parse("e-p7:128:128:128:32-i64:64");
    ASSERT_FALSE(dl.isError()) << dl.getError();
    EXPECT_EQ(dl->getPointerBitWidth(7), 128);
    EXPECT_EQ(dl->getPointerABIAlign(7), 16);
  }

  // An address space with no explicit pointer spec falls back to the default
  // address space (AS 0), matching LLVM semantics, rather than reading
  // uninitialized state.
  {
    auto dl = M::DataLayout::parse("e-p:32:32-p3:64:64-i64:64");
    ASSERT_FALSE(dl.isError()) << dl.getError();
    // AS0 is explicitly 32-bit; AS3 is explicitly 64-bit.
    EXPECT_EQ(dl->getPointerBitWidth(0), 32);
    EXPECT_EQ(dl->getPointerBitWidth(3), 64);
    // AS7 was never specified, so it mirrors the default address space (AS0).
    EXPECT_EQ(dl->getPointerBitWidth(7), 32);
    EXPECT_EQ(dl->getPointerABIAlign(7), 4);
  }

  // Address spaces beyond the tracked range (e.g. x86-64's p270/p271/p272
  // mixed-pointer modes) are skipped rather than rejected, so real target
  // datalayouts still parse and the default address space stays authoritative.
  {
    auto dl = M::DataLayout::parse(
        "e-p:64:64-p270:32:32-p271:32:32-p272:64:64-i64:64");
    ASSERT_FALSE(dl.isError()) << dl.getError();
    EXPECT_EQ(dl->getPointerBitWidth(0), 64);
    EXPECT_EQ(dl->getPointerABIAlign(0), 8);
  }
}
