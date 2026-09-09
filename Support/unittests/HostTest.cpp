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

#include "Support/MArchTarget/Host.h"
#include "Support/Threading/HWInfo.h"
#include "llvm/Support/MemoryBuffer.h"

#include "gtest/gtest.h"

using namespace M;

#if HAVE_LINUX_X86_SYSTEM_INFO
// A two socket system, first with 3 cores + SMT, the second with 2 plain cores.
constexpr static const char *kCpuInfo = R"(
processor	: 0
cpu MHz		: 2200.000
physical id	: 0
siblings	: 6
core id		: 0
cpu cores	: 3
cpuid level	: 16

processor	: 1
cpu MHz		: 2200.000
physical id	: 0
siblings	: 6
core id		: 1
cpu cores	: 3
cpuid level	: 16

processor	: 2
cpu MHz		: 2200.000
physical id	: 0
siblings	: 6
core id		: 2
cpu cores	: 3
cpuid level	: 16

processor	: 3
cpu MHz		: 2200.000
physical id	: 0
siblings	: 6
core id		: 0
cpu cores	: 3
cpuid level	: 16

processor	: 4
cpu MHz		: 2200.000
physical id	: 0
siblings	: 6
core id		: 1
cpu cores	: 3
cpuid level	: 16

processor	: 5
cpu MHz		: 2200.000
physical id	: 0
siblings	: 6
core id		: 2
cpu cores	: 3
cpuid level	: 16

processor	: 6
cpu MHz		: 2200.000
physical id	: 1
siblings	: 2
core id		: 0
cpu cores	: 2
cpuid level	: 16

processor	: 7
cpu MHz		: 2200.000
physical id	: 1
siblings	: 2
core id		: 1
cpu cores	: 2
cpuid level	: 16
)";

TEST(Host, GetLinuxX86CPUSystemInfoImpl) {
  cpu_set_t availCpus;
  CPU_ZERO(&availCpus);
  const size_t numCpus = 8;
  for (size_t i = 0; i < numCpus; ++i) {
    if (i == 3)
      // Disable CPU 3.
      continue;
    CPU_SET(i, &availCpus);
  }

  auto buf = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(kCpuInfo, strlen(kCpuInfo)));
  ErrorOr<CPUSystemInfo> errOr =
      Detail::getLinuxX86CPUSystemInfoImpl(availCpus, std::move(buf));
  EXPECT_FALSE(errOr.isError());
  const CPUSystemInfo &info = errOr.get();

  std::string str;
  llvm::raw_string_ostream os(str);
  os << info;
  EXPECT_EQ(str,
            "CPUSystemInfo(Socket({0}, {1, 4}, {2, 5}), Socket({6}, {7}))");

  ASSERT_EQ(info.sockets.size(), 2UL);
  ASSERT_EQ(info.sockets[0].physicalCores.size(), 3UL);
  EXPECT_EQ(info.sockets[0].physicalCores[0].virtualCores.size(), 1UL);
  EXPECT_EQ(info.sockets[0].physicalCores[1].virtualCores.size(), 2UL);
  EXPECT_EQ(info.sockets[0].physicalCores[2].virtualCores.size(), 2UL);
  ASSERT_EQ(info.sockets[1].physicalCores.size(), 2UL);
  EXPECT_EQ(info.sockets[1].physicalCores[0].virtualCores.size(), 1UL);
  EXPECT_EQ(info.sockets[1].physicalCores[1].virtualCores.size(), 1UL);

  std::vector<size_t> actualCpuIDs = info.getPreferredCpuIDs(9);
  std::vector<size_t> expectedCpuIds = {0, 1, 2,           6,          7,
                                        4, 5, kNoAffinity, kNoAffinity};
  EXPECT_EQ(actualCpuIDs, expectedCpuIds);
}

constexpr static std::string_view kCgroup = R"(
13:pids:/user.slice/user-1000.slice/session-1.scope
12:rdma:/
11:blkio:/user.slice
10:freezer:/
9:devices:/user.slice
8:perf_event:/
7:memory:/user.slice/user-1000.slice/session-1.scope
6:cpuset:/
5:misc:/
4:hugetlb:/
3:net_cls,net_prio:/
2:cpu,cpuacct:/user.slice
1:name=systemd:/user.slice/user-1000.slice/session-1.scope
)";

TEST(Host, ParseV1CPUCgroup) {
  auto buf = llvm::MemoryBuffer::getMemBuffer(kCgroup);
  auto errOrCgroup = Detail::parseV1CPUCgroupFile(*buf);
  EXPECT_FALSE(errOrCgroup.isError());
  EXPECT_EQ(*errOrCgroup, "/user.slice");
}

TEST(Host, ParseV1CPULimits) {
  auto quotaBuf = llvm::MemoryBuffer::getMemBuffer("-1\n");
  auto periodBuf = llvm::MemoryBuffer::getMemBuffer("100001\n");
  auto errOrLimits = Detail::parseV1CPULimits(*quotaBuf, *periodBuf);
  EXPECT_FALSE(errOrLimits.isError());
  EXPECT_EQ(errOrLimits->quota_us, -1);
  EXPECT_EQ(errOrLimits->period_us, 100001);
}

constexpr static StringRef kCgroupV2 = R"(
0::/user.slice/user-1000.slice/session-1.scope
)";

TEST(Host, ParseV2CPUCgroup) {
  auto buf = llvm::MemoryBuffer::getMemBuffer(kCgroupV2);
  {
    auto errOrCgroup =
        Detail::parseV2CPUCgroupFile(*buf, [](StringRef path) { return true; });
    EXPECT_FALSE(errOrCgroup.isError());
    EXPECT_EQ(*errOrCgroup, "/user.slice/user-1000.slice/session-1.scope");
  }
  {
    // Ancestor membership
    size_t i{0};
    auto errOrCgroup = Detail::parseV2CPUCgroupFile(
        *buf, [&](StringRef path) { return ++i == 3; });
    EXPECT_FALSE(errOrCgroup.isError());
    EXPECT_EQ(*errOrCgroup, "/user.slice");
  }
  {
    // Root membership
    size_t i{0};
    auto errOrCgroup = Detail::parseV2CPUCgroupFile(
        *buf, [&](StringRef path) { return ++i == 4; });
    EXPECT_FALSE(errOrCgroup.isError());
    EXPECT_EQ(*errOrCgroup, "");
  }
}

TEST(Host, ParseV2CPUCgroupMissing) {
  auto buf = llvm::MemoryBuffer::getMemBuffer(kCgroupV2);
  auto errOrCgroup =
      Detail::parseV2CPUCgroupFile(*buf, [](StringRef path) { return false; });
  EXPECT_TRUE(errOrCgroup.isError());
}

TEST(Host, ParseV2CPULimits) {
  auto maxBuf = llvm::MemoryBuffer::getMemBuffer("max 100001\n");
  auto errOrLimits = Detail::parseV2CPULimits(*maxBuf);
  EXPECT_FALSE(errOrLimits.isError());
  EXPECT_EQ(errOrLimits->quota_us, -1);
  EXPECT_EQ(errOrLimits->period_us, 100001);
}

TEST(Host, CPULimitsToMillicores) {
  // 128 cores: quota=12800000us, period=100000us -> 128000 millicores.
  // This previously overflowed: 1000 * 12800000 > INT_MAX.
  Detail::linuxCPULimits limits;
  limits.quota_us = 12800000;
  limits.period_us = 100000;
  auto millicores = limits.toMillicores();
  ASSERT_TRUE(millicores.has_value());
  EXPECT_EQ(*millicores, 128000UL);
}

TEST(Host, CPULimitsToMillicoresNoQuota) {
  Detail::linuxCPULimits limits;
  EXPECT_FALSE(limits.toMillicores().has_value());
}

constexpr static StringRef kCgroupHybrid = R"(
13:pids:/user.slice/user-1000.slice/session-1.scope
12:rdma:/
0::/user.slice/user-1000.slice/session-1.scope
)";

TEST(Host, ParseV1CPUCgroupHybridMissing) {
  auto buf = llvm::MemoryBuffer::getMemBuffer(kCgroupHybrid);
  auto errOrCgroup = Detail::parseV1CPUCgroupFile(*buf);
  // CPU controller not mounted as v1.
  EXPECT_TRUE(errOrCgroup.isError());
}

TEST(Host, ParseV2CPUCgroupHybrid) {
  auto buf = llvm::MemoryBuffer::getMemBuffer(kCgroupHybrid);
  auto errOrCgroup =
      Detail::parseV2CPUCgroupFile(*buf, [](StringRef path) { return true; });
  EXPECT_FALSE(errOrCgroup.isError());
  EXPECT_EQ(*errOrCgroup, "/user.slice/user-1000.slice/session-1.scope");
}

TEST(Host, NUMATopologyQueries) {
  // Get NUMATopology instance.
  const auto &errOrTopo = NUMATopology::get();
  ASSERT_FALSE(errOrTopo.isError()) << errOrTopo.getError();
  const NUMATopology &topo = *errOrTopo;

  // Check that getNumaNodes returns at least one NUMA node.
  const auto &numaNodes = topo.getNumaNodes();
  ASSERT_FALSE(numaNodes.empty());

  for (int node : numaNodes) {
    // Check that getCpuIdsForNumaNode returns at least one CPU core.
    auto cpuIds = topo.getCpuIdsForNumaNode(node);
    EXPECT_FALSE(cpuIds.empty());

    // Check that getPciBusesForNumaNode succeeds (may return empty list).
    auto pciBuses = topo.getPciBusesForNumaNode(node);
  }

  // Check that getNumaNodeForPciBus returns the correct NUMA node for a PCI bus
  // address.
  auto pciBuses = topo.getPciBusesForNumaNode(0);
  if (!pciBuses.empty()) {
    const std::string &firstPciBus = pciBuses[0];
    int numaNode = topo.getNumaNodeForPciBus(firstPciBus);
    EXPECT_EQ(numaNode, 0);
  }
}

#endif
