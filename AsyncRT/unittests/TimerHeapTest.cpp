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

#include "AsyncRT/Runtime/TimerHeap.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Runtime/HostSystem.h"

#include "gtest/gtest.h"

using namespace M::AsyncRT;
using std::chrono::nanoseconds;
using std::chrono::steady_clock;
using std::chrono::time_point;

namespace {

struct TimerHeapTest : public testing::Test {
  CPUDeviceRef cpuDevice;
  time_point<steady_clock> start;
  TimerHeap heap;

  TimerHeapTest()
      : cpuDevice(getOrCreateCPUDevice(CPUDeviceSource::Test)),
        start(steady_clock::now()) {}

  AsyncValueRef<Chain> in(int64_t ns) {
    AsyncValueRef<Chain> out = AsyncValueRef<Chain>::allocate(*cpuDevice);
    TimerHeap::deadline expiration = steady_clock::now() + nanoseconds(ns);
    heap.push(expiration, out);
    return out;
  }

  void passed(int64_t ns) {
    EXPECT_GE(steady_clock::now(), start + nanoseconds(ns));
  }
};

TEST_F(TimerHeapTest, Serial) {
  await(in(0));
  await(in(10));
  await(in(100));
  await(in(1'000));
  passed(1'000);
}

TEST_F(TimerHeapTest, OutOfOrder) {
  auto a = in(1'000);
  auto b = in(100'000);    // 100us.
  auto c = in(10'000'000); // 10ms.
  await(c);
  passed(10'000'000);
  await(b);
  await(a);
}

TEST_F(TimerHeapTest, Cancelation) {
  auto a = in(1'000);
  auto b = in(100'000);    // 100us.
  auto c = in(10'000'000); // 10ms.
  heap.cancel(a);
  heap.cancel(b);
  await(c);
  passed(10'000'000);
}

} // namespace
