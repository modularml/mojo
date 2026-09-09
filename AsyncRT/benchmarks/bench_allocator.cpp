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
// Benchmarks the per-allocation cost of the TCMalloc-backed Runtime allocator
// in a tight alloc/free loop. Run with MODULAR_LOG_LEVEL=DEBUG to measure the
// overhead of MLOG_DEBUG calls in the allocate/free hot path.
//
// Usage:
//   ./bench_allocator                         # baseline (no logging)
//   MODULAR_LOG_LEVEL=DEBUG ./bench_allocator # with logging overhead
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/Allocator.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Runtime/HostSystem.h"
#include "Support/MicroBenchmark.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;
using namespace M::AsyncRT;

namespace {

// Number of alloc/free pairs per timed iteration. Large enough to amortize
// timing overhead and produce stable ns/alloc measurements.
constexpr size_t kAllocsPerIter = 1000;
constexpr size_t kWarmupIters = 5;

void runBench(llvm::StringRef name, size_t size, size_t alignment,
              Allocator *allocator) {
  MicroBenchmark bench(name, [&](MicroBenchmark::State &state) {
    for (auto _ : state) {
      for (size_t i = 0; i < kAllocsPerIter; ++i) {
        void *ptr = allocator->allocateBytes(size, alignment);
        MicroBenchmark::doNotOptimizeAway(ptr);
        allocator->deallocateBytes(ptr, size);
      }
    }
  });

  MicroBenchmark::RunOptions opts;
  opts.warmupIterations = kWarmupIters;
  opts.maxBatchSize = 1;

  if (auto err = bench.run(opts); err.isError()) {
    llvm::errs() << name << ": benchmark failed: " << err.takeError() << "\n";
    return;
  }

  MicroBenchmark::ReportOptions reportOpts;
  reportOpts.timeUnit = MicroBenchmark::TimeUnit::kNanoseconds;
  bench.report(llvm::outs(), reportOpts);

  double nsPerAlloc =
      bench.measurement(MicroBenchmark::ReportMetric::kMedianLatency,
                        MicroBenchmark::TimeUnit::kNanoseconds) /
      static_cast<double>(kAllocsPerIter);
  llvm::errs() << "  " << name << ": " << nsPerAlloc << " ns/alloc (median)\n";
}

} // namespace

int main() {
  CPUDeviceOptions opts;
  opts.withTCMallocAllocator();
  CPUDeviceRef cpuDevice = getOrCreateCPUDevice(CPUDeviceSource::Test, opts);
  Allocator *allocator = cpuDevice->getAllocator();

  llvm::errs() << "Allocator benchmark (" << kAllocsPerIter
               << " allocs/iter)\n";
  llvm::errs()
      << "Set MODULAR_LOG_LEVEL=DEBUG to measure logging overhead.\n\n";

  // Small: representative of a MEFExecutor slab (few hundred bytes)
  runBench("alloc_512B", 512, 64, allocator);
  // Medium: small tensor buffer (1 MB)
  runBench("alloc_1MB", 1 << 20, 64, allocator);
  // Large: typical NUMA benchmark tensor (16 MB = 1024 * 4096 * float32)
  runBench("alloc_16MB", 16 << 20, 64, allocator);

  return 0;
}
