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

#include "Support/Profiling/TimeProfiler.h"

#include "Config/Version.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/Globals/Globals.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "Support/MArchTarget/Host.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ios>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace M;
using M::ProfilingDetail::TimeTraceThreadProfiler;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::system_clock;
using std::chrono::time_point;
using std::chrono::time_point_cast;

/// Maximum number of BeginEvent parents to include in final event entry
/// details.
constexpr size_t kMaxParentChain = 2;

//===----------------------------------------------------------------------===//
// ProfilingDetail::Label
//===----------------------------------------------------------------------===//

void ProfilingDetail::Label::intern(StringArena &stringArena) {
  if (tag == kOwned || stringRef.empty())
    return;
  stringRef = intern(stringArena, stringRef);
  tag = kOwned;
}

std::string ProfilingDetail::Label::toString() const {
  std::string str = stringRef.str();
  if (intPayload == kNoIntPayload)
    return str;

  if (str.empty())
    return std::to_string(intPayload);

  str += ":";
  str += std::to_string(intPayload);
  return str;
}

//===----------------------------------------------------------------------===//
// ProfilingDetail::BeginEvent
//===----------------------------------------------------------------------===//

void ProfilingDetail::BeginEvent::dump() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  os << "PROFILE: ";
  os << llvm::format("%8lu", llvm::get_threadid());
  os << llvm::format("  BEG  %16lu  ", id);
  os << name.toString();
  std::string d = detail.toString();
  if (!d.empty())
    os << "/" << d;
  if (parentId)
    os << "  (parent " << parentId << ")";
  os << "\n";
  if (dumpFn)
    os << (*dumpFn)();
  llvm::dbgs() << str;
}

//===----------------------------------------------------------------------===//
// ProfilingDetail::EndEvent
//===----------------------------------------------------------------------===//

void ProfilingDetail::EndEvent::dump() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  os << "PROFILE: ";
  os << llvm::format("%8lu", llvm::get_threadid());
  os << llvm::format("  END  %16lu  \n", id);
  if (dumpFn)
    os << (*dumpFn)();
  llvm::dbgs() << str;
}

//===----------------------------------------------------------------------===//
// ProfilingDetail::SampleEvent
//===----------------------------------------------------------------------===//

void ProfilingDetail::SampleEvent::dump() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  os << "PROFILE: ";
  os << llvm::format("%8lu", llvm::get_threadid());
  os << "  SAM  " << name.toString() << "  " << value << "\n";
  llvm::dbgs() << str;
}

//===----------------------------------------------------------------------===//
// ProfilingDetail::DebugEvent
//===----------------------------------------------------------------------===//

void ProfilingDetail::DebugEvent::dump() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  os << "PROFILE: ";
  os << llvm::format("%8lu", llvm::get_threadid());
  os << "  DBG  " << msg << "\n";
  llvm::dbgs() << str;
}

//===----------------------------------------------------------------------===//
// ProfilingDetail::CompletedEntry
//===----------------------------------------------------------------------===//

static void prepend(const std::string &lhs, std::string &rhs) {
  if (lhs.empty())
    return;
  if (rhs.empty()) {
    rhs = lhs;
    return;
  }
  rhs = lhs + "/" + rhs;
}

void ProfilingDetail::CompletedEntry::mergeEndIntoBegin(
    uint64_t endTid, const EndEvent &endEvent) {
  assert(flavor == kBegin);
  assert(id == endEvent.id);
  tid = endTid;
  end = endEvent.end;
  dur = endEvent.end - start;
}

void ProfilingDetail::CompletedEntry::mergeBeginIntoEnd(
    const CompletedEntry &beginEntry) {
  assert(flavor == kEnd);
  assert(id == beginEntry.id);
  parentId = beginEntry.parentId;
  dur = beginEntry.dur;
  name = beginEntry.name;
  detail = beginEntry.detail;
}

void ProfilingDetail::CompletedEntry::prependParents(
    ArrayRef<const CompletedEntry *> parents) {
  for (auto &parent : parents) {
    prepend(parent->name, name);
    prepend(parent->detail, detail);
  }
}

void ProfilingDetail::CompletedEntry::print(llvm::json::OStream &os,
                                            TimePointType startTime,
                                            llvm::sys::Process::Pid pid,
                                            DurationType granularity) const {
  if (flavor == kEnd || flavor == kDebug)
    return;
  bool isSample = flavor == kSample;
  if (!isSample && dur < granularity)
    return;

  auto startUs = FloatUsType(start - startTime).count();
  auto durUs = FloatUsType(end - start).count();
  os.object([&] {
    os.attribute("pid", pid);
    os.attribute("tid", int64_t(tid));
    os.attribute("ph", isSample ? "C" : "X");
    os.attribute("ts", startUs);
    os.attribute("dur", durUs);
    os.attribute("name", name);
    if (isSample) {
      os.attributeObject("args", [&]() { os.attribute("value", value); });
    } else {
      if (!detail.empty())
        os.attributeObject("args", [&]() { os.attribute("detail", detail); });
    }
  });
}

void ProfilingDetail::CompletedEntry::print(llvm::raw_pwrite_stream &os,
                                            TimePointType startTime) const {
  os << llvm::format("%6d  %10ld  ", tid,
                     duration_cast<microseconds>(start - startTime).count());
  switch (flavor) {
  case kBegin:
    os << "BEG  ";
    break;
  case kEnd:
    os << "END  ";
    break;
  case kSample:
    os << "SAM  ";
    break;
  case kDebug:
    os << "DBG  ";
    break;
  }
  os << llvm::format("%10ld  ", duration_cast<microseconds>(dur).count());
  os << name;
  if (flavor == kSample)
    os << value;
  if (!detail.empty())
    os << "/" << detail;
  os << "\n";
}

//===----------------------------------------------------------------------===//
// ProfilingDetail::TimeTraceThreadProfiler
//===----------------------------------------------------------------------===//

ProfilingDetail::TimeTraceThreadProfiler::TimeTraceThreadProfiler(
    uint16_t threadIndex, uint64_t level)
    : tid(llvm::get_threadid()),
      nextId((static_cast<uint64_t>(threadIndex) << 48) + 1),
      runtimeProfilingTypeMask(level) {
  llvm::get_thread_name(threadName);
}

std::pair<size_t, size_t> ProfilingDetail::TimeTraceThreadProfiler::intern() {
  size_t initSize = stringArena.size();
  beginEvents.enumerate(
      [this](BeginEvent &event) { event.intern(stringArena); });
  sampleEvents.enumerate(
      [this](SampleEvent &event) { event.intern(stringArena); });
  return {initSize, stringArena.size()};
}

//===----------------------------------------------------------------------===//
// ProfilingDetail::ThreadProfilerContext
//===----------------------------------------------------------------------===//

ProfilingDetail::TimeTraceThreadProfiler *
ProfilingDetail::ThreadProfilerContext::get() {
  static thread_local ThreadProfilerContext instance;
  if (!instance.profiler) {
    if (auto *ctx = Globals::getGlobalProfilerContext()) {
      std::lock_guard<std::mutex> lock(ctx->lock);

      // Assign a unique thread index so every begin event id is globally
      // unique.
      uint16_t threadIndex = ctx->nextThreadIndex++;
      assert(ctx->nextThreadIndex > 0 &&
             "too many threads created during profiling");

      // Add this profiler to the main context.
      instance.profiler =
          ctx->profilers
              .emplace_back(std::make_unique<TimeTraceThreadProfiler>(
                  threadIndex, ctx->runtimeProfilingTypeMask))
              .get();
      ctx->threadProfilerContexts.insert(&instance);
    }
  }
  return instance.profiler;
}

ProfilingDetail::ThreadProfilerContext::~ThreadProfilerContext() {
  if (auto *ctx = Globals::getGlobalProfilerContext()) {
    std::lock_guard<std::mutex> lock(ctx->lock);
    ctx->threadProfilerContexts.erase(this);
  }
}

//===----------------------------------------------------------------------===//
// ProfilingDetail::GlobalProfilerContext
//===----------------------------------------------------------------------===//

ProfilingDetail::GlobalProfilerContext::GlobalProfilerContext(
    DurationType granularity, StringRef name, uint64_t level)
    : granularity(granularity), procName(name),
      pid(llvm::sys::Process::getProcessId()),
      beginningOfTime(system_clock::now()), startTime(ClockType::now()),
      runtimeProfilingTypeMask(level) {}

/// Checks that profiler events created with `beginAndPush` have been ended with
/// a corresponding `endAndPop`.
template <typename Iter>
void assertAllProfilerSectionsCompleted(Iter &&derefedProfilers) {
  assert(llvm::all_of(derefedProfilers,
                      [](const TimeTraceThreadProfiler &profiler) {
                        return profiler.stack.empty();
                      }) &&
         "all profiler sections should be ended when calling write");
}

std::vector<ProfilingDetail::CompletedEntry>
ProfilingDetail::GlobalProfilerContext::getCompletedEntries() {
  std::lock_guard<std::mutex> guard(lock);
  auto derefedProfilers = llvm::make_pointee_range(profilers);

  assertAllProfilerSectionsCompleted(derefedProfilers);

  std::unordered_map<ProfilerEventId, CompletedEntry> beginEntryMap;
  std::unordered_map<ProfilerEventId, CompletedEntry> endEntryMap;

  // Collect the BeginEvents.
  for (TimeTraceThreadProfiler &profiler : derefedProfilers) {
    profiler.beginEvents.enumerate([&beginEntryMap](const BeginEvent &event) {
      beginEntryMap.insert({event.id, CompletedEntry(event)});
    });
  }

  // Prepend parent names and descriptions.
  for (auto &pair : beginEntryMap) {
    SmallVector<const CompletedEntry *> parents;
    CompletedEntry *orig = &pair.second;
    CompletedEntry *curr = orig;
    for (size_t i = 0; curr->parentId && i < kMaxParentChain; ++i) {
      auto itr = beginEntryMap.find(curr->parentId);
      if (itr == beginEntryMap.end()) {
        llvm::dbgs() << "PROFILING: WARNING: BeginEvent " << curr->id << " '"
                     << curr->name << "' has invalid parent id "
                     << curr->parentId << "\n";
        break;
      }
      curr = &itr->second;
      parents.push_back(curr);
    }
    if (curr->parentId) {
      llvm::dbgs() << "PROFILING: WARNING: BeginEvent " << orig->id << " '"
                   << orig->name << "' has more than " << kMaxParentChain
                   << " parents\n";
    }
    pair.second.prependParents(parents);
  }

  // Collect the EndEvents, and cross-reference them to the BeginEvents.
  for (TimeTraceThreadProfiler &profiler : derefedProfilers) {
    profiler.endEvents.enumerate([&profiler, &beginEntryMap,
                                  &endEntryMap](const EndEvent &event) {
      auto itr = beginEntryMap.find(event.id);
      if (itr == beginEntryMap.end()) {
        llvm::dbgs() << "PROFILING: WARNING: EndEvent " << event.id
                     << " has no matching BeginEvent\n";
        return;
      }
      itr->second.mergeEndIntoBegin(profiler.tid, event);
      auto itr2 =
          endEntryMap.insert({event.id, CompletedEntry(profiler.tid, event)})
              .first;
      itr2->second.mergeBeginIntoEnd(itr->second);
    });
  }

  // Gather all the completed entries so far.
  std::vector<CompletedEntry> result;
  result.reserve(beginEntryMap.size() + endEntryMap.size());
  for (auto &pair : beginEntryMap)
    result.emplace_back(std::move(pair.second));
  beginEntryMap.clear();
  for (auto &pair : endEntryMap)
    result.emplace_back(std::move(pair.second));
  endEntryMap.clear();

  // Gather the SampleEvents and DebugEvents.
  for (TimeTraceThreadProfiler &profiler : derefedProfilers) {
    profiler.sampleEvents.enumerate(
        [&profiler, &result](const SampleEvent &event) {
          result.emplace_back(profiler.tid, event);
        });
    profiler.debugEvents.enumerate(
        [&profiler, &result](const DebugEvent &event) {
          result.emplace_back(profiler.tid, event);
        });
  }

  // Place everything in total order.
  std::sort(result.begin(), result.end());

  return result;
}

void ProfilingDetail::GlobalProfilerContext::setRuntimeProfilingTypeMask(
    uint64_t typeMask) {
  std::lock_guard<std::mutex> guard(lock);
  runtimeProfilingTypeMask = typeMask;

  auto derefedProfilers = llvm::make_pointee_range(profilers);

  // Sanity check that there are no unfinished profiling segments in flight.
  assertAllProfilerSectionsCompleted(derefedProfilers);

  // Collect begin and end events and check that each is a pair.
  DenseSet<ProfilerEventId> beginEvents;
  DenseSet<ProfilerEventId> endEvents;
  for (TimeTraceThreadProfiler &profiler : derefedProfilers) {
    profiler.beginEvents.enumerate([&beginEvents](const BeginEvent &event) {
      beginEvents.insert(event.id);
    });
  }
  for (TimeTraceThreadProfiler &profiler : derefedProfilers) {
    profiler.endEvents.enumerate(
        [&endEvents](const EndEvent &event) { endEvents.insert(event.id); });
  }
  assert(beginEvents == endEvents && "expected matching begin and end events");

  for (TimeTraceThreadProfiler &profiler : derefedProfilers)
    profiler.runtimeProfilingTypeMask = typeMask;
}

void ProfilingDetail::GlobalProfilerContext::writeJsonTrace(
    llvm::raw_pwrite_stream &os, ArrayRef<CompletedEntry> entries) {
  llvm::json::OStream jsonOS(os);
  jsonOS.objectBegin();
  jsonOS.attributeBegin("traceEvents");
  jsonOS.arrayBegin();

  // Emit all events for the main flame graph.
  for (const auto &entry : entries)
    entry.print(jsonOS, startTime, pid, granularity);

  auto writeMetadataEvent = [&](const char *name, uint64_t tid, StringRef arg) {
    jsonOS.object([&] {
      jsonOS.attribute("cat", "");
      jsonOS.attribute("pid", pid);
      jsonOS.attribute("tid", int64_t(tid));
      jsonOS.attribute("ts", 0);
      jsonOS.attribute("ph", "M");
      jsonOS.attribute("name", name);
      jsonOS.attributeObject("args", [&] { jsonOS.attribute("name", arg); });
    });
  };

  writeMetadataEvent("process_name", pid, procName);
  auto derefedProfilers = llvm::make_pointee_range(profilers);
  for (const TimeTraceThreadProfiler &profile : derefedProfilers)
    writeMetadataEvent("thread_name", profile.tid, profile.threadName);

  jsonOS.arrayEnd();
  jsonOS.attributeEnd();

  // Emit the absolute time when time profiling started. This can be used to
  // combine the profiling data from multiple processes and preserve actual
  // time intervals.
  jsonOS.attribute("beginningOfTime",
                   time_point_cast<microseconds>(beginningOfTime)
                       .time_since_epoch()
                       .count());

  // Emit input tensor info
  jsonOS.attributeBegin("tensorInfo");
  jsonOS.objectBegin();
  jsonOS.attributeBegin("inputShapes");
  jsonOS.arrayBegin();
  for (auto &shape : inputShapes)
    jsonOS.value(shape);
  jsonOS.arrayEnd();
  jsonOS.attributeEnd();
  jsonOS.objectEnd();
  jsonOS.attributeEnd();

  // Emit software version info
  jsonOS.attributeBegin("versionInfo");
  jsonOS.objectBegin();
  ModularVersion version = getModularVersion();
  jsonOS.attribute("modular-git-sha", version.revision);
  jsonOS.attribute("modular-build-type", version.buildType);
  std::ostringstream profilingLevelOctal;
  profilingLevelOctal << std::oct << "0" << MODULAR_ASYNCRT_MAX_PROFILING_LEVEL;
  jsonOS.attribute("modular-profiling-level", profilingLevelOctal.str());
  jsonOS.objectEnd();
  jsonOS.attributeEnd();

  if constexpr (kProfilingEnabled) {
    // Emit the host machine info, if we can retrieve
    // it.
    auto hostMachineInfoOr = getHostMachineInfo();
    if (hostMachineInfoOr.isError()) {
      llvm::dbgs()
          << "PROFILE: WARNING: unable to retrieve system-info for tracefile\n";
    } else {
      jsonOS.attributeBegin("hostMachineInfo");
      hostMachineInfoOr.takeValue().print(jsonOS);
      jsonOS.attributeEnd();
    }
  }

  jsonOS.objectEnd();
  os.flush();
}

void ProfilingDetail::GlobalProfilerContext::writeTextTrace(
    llvm::raw_pwrite_stream &os, ArrayRef<CompletedEntry> entries) {
  os << "Thread   Start(us)  B/E     Dur(us)  Name/Detail\n";
  os << "------  ----------  ---  ----------  "
        "------------------------------\n";
  for (const auto &entry : entries)
    entry.print(os, startTime);
  os.flush();
}

//===----------------------------------------------------------------------===//
// TimeTraceProfiler
//===----------------------------------------------------------------------===//

TimeTraceProfiler::TimeTraceProfiler(unsigned timeTraceGranularity,
                                     StringRef procName, StringRef filename,
                                     uint64_t runtimeProfilingTypeMask) {
  profileFilename = filename.str();

  if constexpr (!ProfilingDetail::kProfilingEnabled) {
    llvm::dbgs() << "PROFILE: INFO: Profiling is not enabled at compile time, "
                    "only direct profiling entries will be captured\n";
  } else {
    llvm::dbgs() << llvm::format(
        "PROFILE: INFO: Recording profiling entries at level 0%o\n",
        MODULAR_ASYNCRT_MAX_PROFILING_LEVEL);
  }

#ifndef NDEBUG
  llvm::dbgs() << "PROFILE: WARNING: Profiling with NDEBUG not defined\n";
#endif
#ifdef MODULAR_DEBUG
  llvm::dbgs() << "PROFILE: WARNING: Profiling with MODULAR_DEBUG defined\n";
#endif

  auto *ctx = new ProfilingDetail::GlobalProfilerContext(
      std::chrono::microseconds(timeTraceGranularity),
      llvm::sys::path::filename(procName), runtimeProfilingTypeMask);
  [[maybe_unused]] auto *prevCtx = Globals::exchangeGlobalProfilerContext(ctx);
  assert(prevCtx == nullptr && "profiler should not be initialized");

  // Prep the profiler for the main thread.
  (void)ProfilingDetail::ThreadProfilerContext::get();
}

TimeTraceProfiler::~TimeTraceProfiler() {
  auto *ctx = Globals::exchangeGlobalProfilerContext(nullptr);
  assert(ctx && "profiler should be initialized");
  // Clear out any dangling pointers in thread profiler contexts.
  {
    std::lock_guard<std::mutex> guard(ctx->lock);
    for (auto *tpc : ctx->threadProfilerContexts)
      tpc->profiler = nullptr;
  }
  delete ctx;
}

void TimeTraceProfiler::addInputShape(const std::string &shape) {
  auto *ctx = Globals::getGlobalProfilerContext();
  assert(ctx && "profiler should be initialized");
  std::lock_guard<std::mutex> guard(ctx->lock);
  ctx->inputShapes.push_back(shape);
}

uint64_t TimeTraceProfiler::runtimeProfilingTypeMask() {
  auto *ctx = Globals::getGlobalProfilerContext();
  assert(ctx && "profiler should be initialized");

  std::lock_guard<std::mutex> guard(ctx->lock);
  return ctx->runtimeProfilingTypeMask;
}

void TimeTraceProfiler::setRuntimeProfilingTypeMask(uint64_t typeMask) {
  auto *ctx = Globals::getGlobalProfilerContext();
  assert(ctx && "profiler should be initialized");

  ctx->setRuntimeProfilingTypeMask(typeMask);
}

ErrorOrSuccess TimeTraceProfiler::write(StringRef fallbackFileName) {
  auto *ctx = Globals::getGlobalProfilerContext();
  assert(ctx && "profiler should be initialized");

  llvm::dbgs() << "PROFILE: INFO: Preparing entries\n";
  std::vector<ProfilingDetail::CompletedEntry> entries =
      ctx->getCompletedEntries();

  // Set up filename base.
  if (profileFilename.empty())
    profileFilename = fallbackFileName == "-" ? "out" : fallbackFileName.str();

  std::error_code ec;

  {
    // Write time trace.
    std::string tracePath = profileFilename == "-"
                                ? profileFilename
                                : profileFilename + ".time-trace";
    llvm::dbgs() << "PROFILE: INFO: Writing " << entries.size()
                 << " entries to JSON " << tracePath << "\n";
    llvm::raw_fd_ostream os(tracePath, ec, llvm::sys::fs::OF_TextWithCRLF);
    if (ec)
      return Error(Twine("could not open ") + tracePath + "(" +
                   Twine(ec.message()) + ")");
    ctx->writeJsonTrace(os, entries);
  }

  {
    // Write the raw event stream.
    std::string eventStreamPath = profileFilename == "-"
                                      ? profileFilename
                                      : profileFilename + ".time-events.txt";
    llvm::dbgs() << "PROFILE: INFO: Writing " << entries.size()
                 << " entries to text " << eventStreamPath << "\n";
    llvm::raw_fd_ostream os(eventStreamPath, ec,
                            llvm::sys::fs::OF_TextWithCRLF);
    if (ec)
      return Error(Twine("could not open ") + eventStreamPath + "(" +
                   Twine(ec.message()) + ")");
    ctx->writeTextTrace(os, entries);
  }

  return success();
}

void TimeTraceProfiler::writeJSONForTesting(llvm::raw_pwrite_stream &os) {
  auto *ctx = Globals::getGlobalProfilerContext();
  assert(ctx && "profiler should be initialized");
  std::vector<ProfilingDetail::CompletedEntry> entries =
      ctx->getCompletedEntries();
  ctx->writeJsonTrace(os, entries);
}

void TimeTraceProfiler::intern() {
  auto *ctx = Globals::getGlobalProfilerContext();
  assert(ctx && "profiler should be initialized");
  ProfilingDetail::TimePointType start = ProfilingDetail::ClockType::now();
  std::lock_guard<std::mutex> guard(ctx->lock);
  size_t initStrings = 0, finalStrings = 0;
  llvm::for_each(ctx->threadProfilerContexts,
                 [&initStrings,
                  &finalStrings](ProfilingDetail::ThreadProfilerContext *ctx) {
                   auto [init, final] = ctx->profiler->intern();
                   initStrings += init;
                   finalStrings += final;
                 });
  ProfilingDetail::TimePointType end = ProfilingDetail::ClockType::now();
  llvm::dbgs() << llvm::format(
      "PROFILE: INFO: %ld strings were copied/moved during profiling. "
      "An additional %ld strings were interned, taking %ldus.\n",
      initStrings, finalStrings - initStrings,
      duration_cast<microseconds>(end - start).count(), initStrings);
}
