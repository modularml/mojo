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

#include "Support/Threading/ThreadAffinity.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/Threading/HWInfo.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/DebugLog.h"
#include <cstddef>

#define DEBUG_TYPE "thread-affinity"

#ifdef __linux__
#define HAVE_LINUX_SET_AFFINITY 1
#include <linux/mempolicy.h>
#include <sys/syscall.h>
#include <unistd.h>
#else
#define HAVE_LINUX_SET_AFFINITY 0
#endif

using namespace M;

//===----------------------------------------------------------------------===//
// Thread affinity
//===----------------------------------------------------------------------===//

#if HAVE_LINUX_SET_AFFINITY
static ErrorOrSuccess setThreadAffinityLinux(size_t cpuID) {
  // Return without setting affinity if CPU core ID is invalid.
  if (cpuID == kNoAffinity)
    return Error("unable to set thread affinity, invalid CPU id");

  // Resolve the NUMA node for the CPU core, if topology is available.
  int numaNode = kAnyNumaNode;
  const ErrorOr<NUMATopology> &topologyOr = NUMATopology::get();
  if (!topologyOr.isError())
    numaNode = topologyOr->getNumaNodeForCpuId(cpuID);

  // Set thread execution affinity to the specified CPU core ID.
  assert(cpuID < CPU_SETSIZE);
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpuID, &cpuset);
  if (int rc = sched_setaffinity(0, sizeof(cpuset), &cpuset))
    return Error("unable to set thread execution affinity: " +
                 std::to_string(rc));

  // Set thread memory policy.
  if (numaNode != kAnyNumaNode) {
    assert(numaNode >= 0 &&
           static_cast<size_t>(numaNode) < sizeof(unsigned long) * 8);
    unsigned long nodemask = 1UL << numaNode;
    if (long rc = syscall(SYS_set_mempolicy, MPOL_PREFERRED, &nodemask,
                          sizeof(nodemask) * 8))
      return Error("unable to set thread memory policy: " + std::to_string(rc));
  }

  return success();
}

static ErrorOrSuccess
runWithThreadAffinityLinux(size_t cpuID, llvm::function_ref<void()> &workFn) {
  // Return without setting affinity if CPU core ID is invalid.
  if (cpuID == kNoAffinity)
    return Error("unable to set thread affinity, invalid CPU id");

  // Resolve the NUMA node for the CPU core, if topology is available.
  int numaNode = kAnyNumaNode;
  const ErrorOr<NUMATopology> &topologyOr = NUMATopology::get();
  if (!topologyOr.isError())
    numaNode = topologyOr->getNumaNodeForCpuId(cpuID);

  // Save and set thread execution affinity.
  cpu_set_t origCpuSet;
  assert(cpuID < CPU_SETSIZE);
  if (int rc = sched_getaffinity(0, sizeof(origCpuSet), &origCpuSet))
    return Error("unable to get thread execution affinity: " +
                 std::to_string(rc));
  cpu_set_t cpuSet;
  CPU_ZERO(&cpuSet);
  CPU_SET(cpuID, &cpuSet);
  if (int rc = sched_setaffinity(0, sizeof(cpuSet), &cpuSet))
    return Error("unable to set thread execution affinity: " +
                 std::to_string(rc));

  // Save and set thread memory policy.
  int origMemPolicy = 0;
  unsigned long origNodeMask = 0;
  if (numaNode != kAnyNumaNode) {
    assert(numaNode >= 0 &&
           static_cast<size_t>(numaNode) < sizeof(unsigned long) * 8);
    if (long rc = syscall(SYS_get_mempolicy, &origMemPolicy, &origNodeMask,
                          sizeof(origNodeMask) * 8, nullptr, 0))
      return Error("unable to get thread memory policy: " + std::to_string(rc));
    unsigned long nodeMask = 1UL << numaNode;
    if (long rc = syscall(SYS_set_mempolicy, MPOL_PREFERRED, &nodeMask,
                          sizeof(nodeMask) * 8))
      return Error("unable to set thread memory policy: " + std::to_string(rc));
  }

  // Assuming -fno-exceptions so no need for exception handling here.
  workFn();

  // Restore thread execution affinity.
  if (int rc = sched_setaffinity(0, sizeof(origCpuSet), &origCpuSet))
    LDBG() << "runWithThreadAffinityLinux: unable to restore "
              "thread execution affinity: "
           << rc;

  // Restore thread memory policy.
  if (numaNode != kAnyNumaNode) {
    long rc = 0;
    if (origMemPolicy == MPOL_DEFAULT)
      rc = syscall(SYS_set_mempolicy, MPOL_DEFAULT, nullptr, 0);
    else
      rc = syscall(SYS_set_mempolicy, origMemPolicy, &origNodeMask,
                   sizeof(origNodeMask) * 8);
    if (rc != 0)
      LDBG() << "runWithThreadAffinityLinux: unable to restore "
                "thread memory policy: "
             << rc;
  }

  return success();
}
#endif

bool M::haveThreadAffinity() {
#if HAVE_LINUX_SET_AFFINITY
  return true;
#else
  return false;
#endif // HAVE_LINUX_SET_AFFINITY
}

ErrorOrSuccess M::setThreadAffinity(size_t cpuID) {
#if HAVE_LINUX_SET_AFFINITY
  return setThreadAffinityLinux(cpuID);
#else
  return Error("setThreadAffinity is not supported by this build");
#endif // HAVE_LINUX_SET_AFFINITY
}

ErrorOrSuccess M::runWithThreadAffinity(size_t cpuID,
                                        llvm::function_ref<void()> workFn) {
#if HAVE_LINUX_SET_AFFINITY
  return runWithThreadAffinityLinux(cpuID, workFn);
#else
  return Error("runWithThreadAffinity is not supported by this build");
#endif // HAVE_LINUX_SET_AFFINITY
}
