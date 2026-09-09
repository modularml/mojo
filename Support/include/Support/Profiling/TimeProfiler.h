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
// This provides lightweight and dependency-free machinery to trace execution
// time around arbitrary code. Three API flavors are available.
//
// The primary API uses a RAII object to trigger tracing:
//
// \code
//   {
//     TimeTraceScope scope(ProfilerEntry<true>::create("my_event_name"));
//     ...my code...
//   }
// \endcode
//
// If the code to be profiled does not have a natural lexical scope then
// it is also possible to start and end events with respect to an implicit
// per-thread stack of profiling entries:
//
// \code
//   ProfilerEntry<true>::createAndPush("my_event_name");
//   ...my code...
//   ProfilerEntry<true>::endAndPop();  // must be called on all control flow
//   paths
// \endcode
//
// Finally, it is also possible to manually create and complete time profiling
// entries. This API allows an entry to be created in one context, stored,
// then completed in another. The completing context need not be on the same
// thread as the creating context:
//
// \code
//   auto entry = ProfilerEntry<true>::create("my_event_name");
//   ...
//   // Possibly on a different thread
//   ...my code...
//   std::move(entry).record();
// \endcode
//
// Time profiling entries can be given an arbitrary name and, optionally,
// an arbitrary 'detail' string.
//
// The main process should first construct a TimeTraceProfiler, which is used
// to anchor the various timing functionality, and exposes support for writing
// to various formats. Note that only one such profiler may be active at any
// given time.
//
// Timestamps come from std::chrono::high_resolution_clock, so all threads
// see the same time at the highest available resolution.
//
// Currently, there are a number of compatible viewers:
//  - http://ui.perfetto.dev is the Chrome profile viewer, under active
//    development by Google as part of the 'Perfetto' project.
//  - https://www.speedscope.app/ has also been reported as an option.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_PROFILING_TIMEPROFILER_H
#define SUPPORT_PROFILING_TIMEPROFILER_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <ratio>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace llvm {
class raw_pwrite_stream;
namespace json {
class OStream;
}
} // namespace llvm

/// Define to 1 to have all enabled profiling events dumped to stderr as they
/// are created. The profiling entry type must be enabled, and profiling must
/// be active. This can help track down hangs/crashes.
#define TRACE_IN_REAL_TIME 0

/// Define to 1 to dump strings which are copied/moved during profiling.
/// These can cause the profiling entries to be slower than the code being
/// profiled!
#define WARN_ABOUT_TRACE_COPIES 0

namespace M {

static constexpr bool kIsProfilingEnabled =
    MODULAR_ASYNCRT_MAX_PROFILING_LEVEL > 0;

// The cmake arg -DMODULAR_ASYNCRT_MAX_PROFILING_LEVEL is interpreted as chunks
// of octal numbers. The TraceType enum indicates which chunk corresponds to
// which type of profiling. Each chunk, representing a type of profiling, has 3
// bits worth of profiling levels to use. For example, to enable level 1 AsyncRT
// profiling but not DEFAULT profiling, you would pass
// -DMODULAR_ASYNCRT_MAX_PROFILING_LEVEL=010 to cmake. To enable level 2 DEFAULT
// profiling, but not AsyncRT profiling, you would pass
// -DMODULAR_ASYNCRT_MAX_PROFILING_LEVEL=02 to cmake.
// This is intended to avoid cluttering profiles with unwanted information by
// giving the user more control over which types of traces they want to see.
struct Trace {
  enum Type : uint8_t {
    // kOther:
    //   For traces not covered by any of the following.
    // Level 1:
    //   DriverProfilerEntry for mt's execution.
    //   ATENProfilerEntry for fallback ATen primitives.
    //   RuntimeCacheProfilerEntry for cache transforms.
    kOther = 0,

    // kAsyncRT:
    //   For traces related to core AsyncRT scheduling and cpuDevice.
    // Level 2:
    //   AlgorithmProfilerEntry for addTask via parallelization helpers
    //   InternalProfilerEntry for tracking thread spinning and sleeping.
    // Level 3:
    //   AllWorkItemsProfilerEntry for the execution of every work/wait item.
    kAsyncRT = 1,

    // kMem:
    //   For traces related to memory usage.
    // Level 1:
    //   MemCopyProfilerEntry for memcpy.
    // Level 2:
    //   MemAllocFreeProfilerEntry for alloc and free.
    // Level 3:
    //   BufferRefLifetimeProfilerEntry for all buffer lifetimes.
    //   MemProfilerEntry for outstanding bytes allocated.
    kMem = 2,

    // kMojo:
    //   For traces related to mojo kernels. Controlled from Mojo side only.
    kMojo = 3,

    // kPrimitives:
    //   For traces related to GraphRT executor.
    // Level 1:
    //   PrimitiveProfilerEntry for all GraphRT primitives.
    kPrimitives = 4,

    // kCompiler:
    //   For traces related to compilation.
    // Level 1:
    //   InterpreterProfilerEntry for Mojo interpreter traces.
    // Level 2:
    //   CacheProfilerEntry for cache-related transforms.
    //   CompilerProfilerEntry for Mojo compiler passes.
    // Level 3:
    //   VerboseCompilerProfilerEntry for very detailed Mojo compiler profiling.
    kCompiler = 5,
  };

  // Each profiling type has 3 bits worth of levels to use
  static constexpr uint64_t kProfilingTypeWidthBits = 3;
  static constexpr uint64_t kProfilingTypeBitmask =
      ((1 << kProfilingTypeWidthBits) - 1);
  static constexpr uint64_t kFullyEnabled =
      std::numeric_limits<uint64_t>::max();
  static constexpr uint64_t typeBitshift(Type type) {
    return type * kProfilingTypeWidthBits;
  }
  static constexpr uint64_t disableMaskType(uint64_t bitMask, Type type) {
    return bitMask & ~(kProfilingTypeBitmask << typeBitshift(type));
  }

  static constexpr bool EnableTrace(Type type, uint64_t level) {
    return level <=
           ((MODULAR_ASYNCRT_MAX_PROFILING_LEVEL >> typeBitshift(type)) &
            kProfilingTypeBitmask);
  }
};

/// Function that returns arbitrary messages when profiler events occur.
/// `TimeTraceScope` has to store a `ProfilerDumpFn` in order to
/// call that function when creating an `EndEvent` if `TRACE_IN_REAL_TIME` is 1.
/// In order to ensure its captures live until the end of the event,
/// `ProfilerDumpFn` has to be a `unique_function`.
using ProfilerDumpFn = llvm::unique_function<std::string() const>;

/// Function to call to return profiling entry name or description string.
/// This uses a `function_ref` because the function will be called immediately
/// upon creation of a `Label` name for a profiling entry.
/// Hence `ProfilerPrintFn`s are not stored and can be a `function_ref`.
using ProfilerPrintFn = llvm::function_ref<std::string()>;

/// Globally unique id for every CreateEvent and SampleEvent. Zero denotes
/// no-event.
using ProfilerEventId = size_t;

/// An InternableString is a StringRef with underlying lifetime guaranteed
/// at least up until the next call to TimeTraceProfiler::intern(). A typical
/// example would be a string attribute from a MEFFile.
struct InternableString : StringRef {
  using StringRef::StringRef;
};

namespace ProfilingDetail {

// Make sure there's no struct slicing with our internal representation.
static_assert(sizeof(StringLiteral) == sizeof(StringRef));
static_assert(sizeof(InternableString) == sizeof(StringRef));

constexpr bool kProfilingEnabled = MODULAR_ASYNCRT_MAX_PROFILING_LEVEL > 0;

// We use the high_resolution_clock for maximum precision.
// It may not be steady (ClockType::is_steady may be false), which means
// it is possible for profiles to yield invalid durations during leap
// second transitions or other system clock adjustments. This rare glitch
// seems worthwhile in exchange for the precision.
// Under linux glibc++ the high_resolution_clock is consistent across threads
// which is necessary for building cross-thread entries.
// It is unknown whether that's the case under Windows, and the C++ standard
// does not appear to impose any thread consistency on any of the clocks.
using ClockType = std::chrono::high_resolution_clock;
using TimePointType = std::chrono::time_point<ClockType>;
using FloatUsType = std::chrono::duration<double, std::micro>;
using DurationType = std::chrono::duration<ClockType::rep, ClockType::period>;

//===----------------------------------------------------------------------===//
// ProfilingDetail::BlockList
//===----------------------------------------------------------------------===//

constexpr size_t kBlockListBlockSize = 1024;

template <typename T>
struct BlockListEntry {
  std::unique_ptr<BlockListEntry> tail;
  SmallVector<T, kBlockListBlockSize> entries;
};

/// Storage for recorded events and strings owned by a per-thread profiler.
template <typename T>
struct BlockList {
  std::unique_ptr<BlockListEntry<T>> head;
  BlockListEntry<T> *last = nullptr;
  size_t totalSize = 0;

  size_t size() const { return totalSize; }

  template <typename... Args>
  const T &emplace_back(Args &&...args) {
    if (head == nullptr) {
      assert(last == nullptr);
      head = std::make_unique<BlockListEntry<T>>();
      last = head.get();
    }
    if (last->entries.size() >= kBlockListBlockSize) {
      assert(last->tail == nullptr);
      last->tail = std::make_unique<BlockListEntry<T>>();
      last = last->tail.get();
    }
    ++totalSize;
    return last->entries.emplace_back(std::forward<Args>(args)...);
  }

  void enumerate(llvm::function_ref<void(const T &)> func) const {
    BlockListEntry<T> *curr = head.get();
    while (curr) {
      llvm::for_each(curr->entries, func);
      curr = curr->tail.get();
    }
  }

  void enumerate(llvm::function_ref<void(T &)> func) {
    BlockListEntry<T> *curr = head.get();
    while (curr) {
      llvm::for_each(curr->entries, func);
      curr = curr->tail.get();
    }
  }
};

//===----------------------------------------------------------------------===//
// ProfilingDetail::Label
//===----------------------------------------------------------------------===//

/// Use an unordered_set as `StringArena`.
/// This is because the majority of profiler events' names are static strings
/// and the ratio of events to names is large.
/// NOTE: `Label::intern()` returns a pointer into this `StringArena`.
/// So it is important that those pointers are still valid after further
/// insertion into the `StringArena`.
/// `unordered_set` satisfies this property, but note that for example
/// `llvm::DenseSet` does not.
using StringArena = std::unordered_set<std::string>;

/// A way to derive the name or detail string for a profiling entry. When safe,
/// allows string copies to be avoided until either the final profile is being
/// written, or a call is made to TimeTraceProfiler::intern().
///
/// Since it is so common, also allows the combination of a string and a
/// single uint64_t value, rendered as "name:42", without requiring the caller
/// to do any string manipulation.
///
/// The following forms of 'strings' are supported:
///   - StringLiteral (subclass of StringRef): Assume the reference will remain
///     valid up to the next call to TimeTraceProfiler::intern().
///   - InternableString (subclass of StringRef): As for StringLiteral.
///   - StringRef: Copy the string reference data into the string arena.
///   - std::string: Move the string into the string arena.
class Label {
public:
  static constexpr int kIntPayloadBits = 62;
  static constexpr uint64_t kNoIntPayload = (1ull << kIntPayloadBits) - 1;

  /*implicit*/
  constexpr Label(uint64_t value = kNoIntPayload)
      : tag(kLiteral), intPayload(value) {
    assert(intPayload == value && "overflow for int payload");
  }

  // CAUTION: No string copy, but may need to be interned.
  // Note we don't treat StringLiteral any differently from InternableString
  // since it is possible for a literal to point to the data segment of a dylib
  // which can be unloaded.
  /*implicit*/
  constexpr Label(StringArena &stringArena, StringLiteral stringLiteral,
                  uint64_t value = kNoIntPayload)
      : stringRef(stringLiteral), tag(kLiteral), intPayload(value) {
    assert(intPayload == value && "overflow for int payload");
  }

  // CAUTION: No string copy, but may need to be interned.
  /*implicit*/
  constexpr Label(StringArena &stringArena, InternableString internableString,
                  uint64_t value = kNoIntPayload)
      : stringRef(internableString), tag(kInternable), intPayload(value) {
    assert(intPayload == value && "overflow for int payload");
  }

  // CAUTION: Copies string into arena.
  /*implicit*/
  Label(StringArena &stringArena, StringRef stringRef,
        uint64_t value = kNoIntPayload)
      : stringRef(intern(stringArena, stringRef)), tag(kOwned),
        intPayload(value) {
#if WARN_ABOUT_TRACE_COPIES
    llvm::errs() << "PROFILE: WARNING: Copied profiling string: "
                 << this->stringRef << "\n";
#endif
    assert(intPayload == value && "overflow for int payload");
  }

  // Moves string.
  Label(StringArena &stringArena, std::string string,
        uint64_t value = kNoIntPayload)
      : stringRef(intern(stringArena, std::move(string))), tag(kOwned),
        intPayload(value) {
#if WARN_ABOUT_TRACE_COPIES
    llvm::errs() << "PROFILE: WARNING: Moved profiling string: "
                 << this->stringRef << "\n";
#endif
    assert(intPayload == value && "overflow for int payload");
  }

  // Calls function and moves string.
  /*implicit*/
  Label(StringArena &stringArena, ProfilerPrintFn printFn,
        uint64_t value = kNoIntPayload)
      : stringRef(intern(stringArena, printFn())), tag(kOwned),
        intPayload(value) {
#if WARN_ABOUT_TRACE_COPIES
    llvm::errs()
        << "PROFILE: WARNING: Moved profiling string from detail function: "
        << this->stringRef << "\n";
#endif
    assert(intPayload == value && "overflow for int payload");
  }

  ~Label() = default;

  /// If the label possibly contains a borrow, evaluate it to its owning
  /// std::string form.
  void intern(StringArena &stringArena);

  /// Returns the label in string form.
  std::string toString() const;

  bool empty() const {
    return intPayload == kNoIntPayload && stringRef.empty();
  }

  std::optional<uint64_t> getInt() const {
    return intPayload < kNoIntPayload ? intPayload : std::optional<uint64_t>();
  }

  // No copy, only move.
  Label(const Label &) = delete;
  Label &operator=(const Label &) = delete;
  Label(Label &&that) = default;
  Label &operator=(Label &&that) = default;

private:
  /// Copies data of stringRef into stringArena and returns a StringRef to it.
  static StringRef intern(StringArena &stringArena, StringRef stringRef) {
    return StringRef(*stringArena.insert(stringRef.str()).first);
  }

  // Moves string into stringArena and returns a StringRef to it.
  static StringRef intern(StringArena &stringArena, std::string string) {
    return StringRef(*stringArena.insert(std::move(string)).first);
  }

  /// Reference to string, which we are ether borrowing, or which resides in
  /// the stringArena of a per-thread profiler.
  StringRef stringRef;

  /// How to interpret the above string reference.
  enum Tag { kLiteral = 0, kInternable = 1, kOwned = 2 };
  uint64_t tag : 2;

  /// Additional integer value, or kNoIntPayload if none.
  uint64_t intPayload : kIntPayloadBits;
};

//===----------------------------------------------------------------------===//
// ProfilingDetail::BeginEvent
//===----------------------------------------------------------------------===//

/// Represents the creation of a timing profiling entry.
struct BeginEvent {
  uint64_t seqNum;
  ProfilerEventId id;
  ProfilerEventId parentId = 0;
  TimePointType start = ClockType::now();
  Label name;
  Label detail;
  std::optional<ProfilerDumpFn> dumpFn;

  template <typename NameStr>
  BeginEvent(StringArena &stringArena, uint64_t seqNum, ProfilerEventId id,
             ProfilerEventId parentId, NameStr &&name,
             uint64_t nameValue = Label::kNoIntPayload,
             std::optional<ProfilerDumpFn> extraDumpFn = std::nullopt)
      : seqNum(seqNum), id(id), parentId(parentId),
        name(stringArena, std::forward<NameStr>(name), nameValue),
        dumpFn(std::move(extraDumpFn)) {}

  template <typename NameStr, typename DetailStr>
  BeginEvent(StringArena &stringArena, uint64_t seqNum, ProfilerEventId id,
             ProfilerEventId parentId, NameStr &&name, DetailStr &&detail,
             uint64_t detailValue = Label::kNoIntPayload)
      : seqNum(seqNum), id(id), parentId(parentId),
        name(stringArena, std::forward<NameStr>(name)),
        detail(stringArena, std::forward<DetailStr>(detail), detailValue) {}

  /// Intern the name and detail labels.
  void intern(StringArena &stringArena) {
    name.intern(stringArena);
    detail.intern(stringArena);
  }

  void dump() const;
};

//===----------------------------------------------------------------------===//
// ProfilingDetail::EndEvent
//===----------------------------------------------------------------------===//

/// Represents the end of a timing profiling entry. It is valid for
/// a profiling entry to begin on one thread and end on another.
struct EndEvent {
  uint64_t seqNum;
  ProfilerEventId id;
  TimePointType end = ClockType::now();
  std::optional<ProfilerDumpFn> dumpFn;

  EndEvent(uint64_t seqNum, ProfilerEventId id,
           std::optional<ProfilerDumpFn> extraDumpFn = std::nullopt)
      : seqNum(seqNum), id(id), dumpFn(std::move(extraDumpFn)) {}

  void dump() const;
};

//===----------------------------------------------------------------------===//
// ProfilingDetail::SampleEvent
//===----------------------------------------------------------------------===//

/// Represents the sampling of an integer value.
struct SampleEvent {
  uint64_t seqNum;
  TimePointType stamp = ClockType::now();
  uint64_t value;
  Label name;

  template <typename NameStr>
  SampleEvent(StringArena &stringArena, uint64_t seqNum, uint64_t value,
              NameStr &&name, uint64_t nameValue = Label::kNoIntPayload)
      : seqNum(seqNum), value(value),
        name(stringArena, std::forward<NameStr>(name), nameValue) {}

  /// Intern the name label.
  void intern(StringArena &stringArena) { name.intern(stringArena); }

  void dump() const;
};

//===----------------------------------------------------------------------===//
// ProfilingDetail::DebugEvent
//===----------------------------------------------------------------------===//

/// Represents a debug entry.
struct DebugEvent {
  uint64_t seqNum;
  TimePointType stamp = ClockType::now();
  std::string msg;

  DebugEvent(uint64_t seqNum, std::string msg)
      : seqNum(seqNum), msg(std::move(msg)) {}

  void dump() const;
};

//===----------------------------------------------------------------------===//
// ProfilingDetail::CompletedEntry
//===----------------------------------------------------------------------===//

/// A completed profiling entry, built from the combination of begin, end,
/// sampling and parent events.
struct CompletedEntry {
  enum Flavor { kBegin = 0, kEnd = 1, kSample = 2, kDebug = 3 };
  Flavor flavor = kBegin;
  uint64_t seqNum = 0;
  ProfilerEventId id = 0;
  ProfilerEventId parentId = 0;
  uint64_t tid = 0;
  TimePointType start;
  TimePointType end;
  DurationType dur{0};
  std::string name;
  std::string detail;
  uint64_t value = 0;

  CompletedEntry() = default;

  explicit CompletedEntry(const BeginEvent &beginEvent)
      : flavor(kBegin), seqNum(beginEvent.seqNum), id(beginEvent.id),
        parentId(beginEvent.parentId), start(beginEvent.start),
        name(beginEvent.name.toString()), detail(beginEvent.detail.toString()) {
  }

  CompletedEntry(uint64_t tid, const EndEvent &endEvent)
      : flavor(kEnd), seqNum(endEvent.seqNum), id(endEvent.id), tid(tid),
        start(endEvent.end), end(endEvent.end) {}

  CompletedEntry(uint64_t tid, const SampleEvent &sampleEvent)
      : flavor(kSample), seqNum(sampleEvent.seqNum), tid(tid),
        start(sampleEvent.stamp), end(sampleEvent.stamp),
        name(sampleEvent.name.toString()), value(sampleEvent.value) {}

  CompletedEntry(uint64_t tid, const DebugEvent &debugEvent)
      : flavor(kDebug), seqNum(debugEvent.seqNum), tid(tid),
        start(debugEvent.stamp), end(debugEvent.stamp), name(debugEvent.msg) {}

  /// Update this begin entry with details from end event.
  void mergeEndIntoBegin(uint64_t endTid, const EndEvent &endEvent);

  /// Update this end entry with details from begin entry.
  void mergeBeginIntoEnd(const CompletedEntry &beginEntry);

  /// Update this entry to include the name and details from all of parents.
  void prependParents(ArrayRef<const CompletedEntry *> parents);

  /// Global temporal ordering, with ties broken by per-thread sequence numbers.
  bool operator<(const CompletedEntry &that) const {
    return std::tie(start, tid, seqNum) <
           std::tie(that.start, that.tid, seqNum);
  }

  /// Prints entry in JSON form to os.
  void print(llvm::json::OStream &os, TimePointType startTime,
             llvm::sys::Process::Pid pid, DurationType granularity) const;

  /// Prints entry in compact form to os.
  void print(llvm::raw_pwrite_stream &os, TimePointType startTime) const;
};

//===----------------------------------------------------------------------===//
// ProfilingDetail::TimeTraceThreadProfiler
//===----------------------------------------------------------------------===//

using BeginEventList = BlockList<BeginEvent>;
using EndEventList = BlockList<EndEvent>;
using SampleEventList = BlockList<SampleEvent>;
using DebugEventList = BlockList<DebugEvent>;

struct TimeTraceThreadProfiler {
  explicit TimeTraceThreadProfiler(uint16_t threadIndex, uint64_t level);

  /// Begin a new timing entry, and return its globally unique id.
  template <typename... Args>
  ProfilerEventId begin(Args &&...args) {
    ProfilerEventId id = nextId++;
    [[maybe_unused]] const BeginEvent &event = beginEvents.emplace_back(
        stringArena, nextSeqNum++, id, /*parentId=*/(ProfilerEventId)0,
        std::forward<Args>(args)...);
#if TRACE_IN_REAL_TIME
    event.dump();
#endif
    return id;
  }

  /// Begin a new timing entry with the given parent, and return its globally
  /// unique id.
  template <typename... Args>
  ProfilerEventId beginWithParent(ProfilerEventId parentId, Args &&...args) {
    ProfilerEventId id = nextId++;
    [[maybe_unused]] const BeginEvent &event = beginEvents.emplace_back(
        stringArena, nextSeqNum++, id, parentId, std::forward<Args>(args)...);
#if TRACE_IN_REAL_TIME
    event.dump();
#endif
    return id;
  }

  /// End the timing entry with the given id. The event need not have been
  /// begun on this thread.
  template <typename... Args>
  void end(ProfilerEventId id, Args &&...args) {
    [[maybe_unused]] const EndEvent &event =
        endEvents.emplace_back(nextSeqNum++, id, std::forward<Args>(args)...);
#if TRACE_IN_REAL_TIME
    event.dump();
#endif
  }

  /// Begin a new timing entry, and push it onto the stack of currently
  /// running entries. A corresponding call to endAndPop() must be made
  /// from the same thread.
  template <typename... Args>
  void beginAndPush(Args &&...args) {
    ProfilerEventId id = begin(std::forward<Args>(args)...);
    stack.push_back(id);
  }

  /// Begin a new timing entry, and push it onto the stack of currently
  /// running entries. A corresponding call to endAndPop() must be made
  /// from the same thread.
  template <typename... Args>
  void beginWithParentAndPush(ProfilerEventId parentId, Args &&...args) {
    ProfilerEventId id = beginWithParent(parentId, std::forward<Args>(args)...);
    stack.push_back(id);
  }

  /// End the most recently pushed timing event.
  void endAndPop() {
    assert(!stack.empty() && "unbalanced push/pop");
    end(stack.pop_back_val());
  }

  ProfilerEventId getCurrentId() const { return currentId; }
  void setCurrentId(ProfilerEventId id) { currentId = id; }

  /// Record a sampling entry.
  template <typename... Args>
  void sample(uint64_t value, Args &&...args) {
    [[maybe_unused]] const SampleEvent &event = sampleEvents.emplace_back(
        stringArena, nextSeqNum++, value, std::forward<Args>(args)...);
#if TRACE_IN_REAL_TIME
    event.dump();
#endif
  }

  /// Record a debugging entry.
  void debug(std::string msg) {
    const DebugEvent &event =
        debugEvents.emplace_back(nextSeqNum++, std::move(msg));
    event.dump();
  }

  /// Checks if the trace type has been disabled at runtime.
  bool isEnabled(Trace::Type type) const {
    return (runtimeProfilingTypeMask >> Trace::typeBitshift(type)) &
           Trace::kProfilingTypeBitmask;
  }

  /// Intern all event labels, returning initial and final number of strings
  /// in the string arena.
  std::pair<size_t, size_t> intern();

  /// The id of the thread this profiler is running on.
  const uint64_t tid;

  /// The name of the thread this profiler is running on.
  SmallString<0> threadName;

  /// Next available begin event id.
  ProfilerEventId nextId;

  /// The next sequence number to use for all events. This helps us preserve
  /// the per-thread temporal ordering of events even when events have the
  /// same start time.
  uint64_t nextSeqNum = 0;

  /// The stack of pushed but not yet popped BeginEvent ids.
  SmallVector<ProfilerEventId> stack;

  /// Recorded events.
  BeginEventList beginEvents;
  EndEventList endEvents;
  SampleEventList sampleEvents;
  DebugEventList debugEvents;

  /// String arena.
  StringArena stringArena;

  /// The 'current' active event id. This can be used to implicitly propagate
  /// a 'parent' event id into a child event. However, great care must be
  /// taken to reset this value to prevent erroneous association of parents
  /// to unrelated children which just happen to run on the same thread.
  ProfilerEventId currentId = 0;

  /// Runtime configurable filter for profiling types (`Trace::Type`).
  /// Currently this only takes "type" into account and ignores "level".
  /// So any non-zero value enables the level, in other words `11111` and
  /// `22222` and `12121` all have the same effect. Set this in Runtime's ctor
  /// via CPUDeviceOptions.runtimeProfilingTypeMask.
  ///
  /// For example:
  ///
  /// AsyncRT::CPUDeviceOptions rtOpt;
  /// rtOpt.runtimeProfilingTypeMask = 1 << Trace::typeBitshift(Trace::kOther);
  /// auto rt = AsyncRT::getOrCreateCPUDevice(AsyncRT::CPUDeviceSource::Test,
  /// rtOpt);
  ///
  /// Creates a Runtime that will only record `kOther` type events.
  uint64_t runtimeProfilingTypeMask;
};

//===----------------------------------------------------------------------===//
// ProfilingDetail::ThreadProfilerContext
//===----------------------------------------------------------------------===//

/// This class represents the profiler context for a specific thread.
struct ThreadProfilerContext {
  ~ThreadProfilerContext();

  /// Return the profiler instance for this thread, or nullptr if profiling
  /// is not active.
  static TimeTraceThreadProfiler *get();

  /// The profiler attached to this thread.
  TimeTraceThreadProfiler *profiler = nullptr;
};

//===----------------------------------------------------------------------===//
// ProfilingDetail::GlobalProfilerContext
//===----------------------------------------------------------------------===//

/// This class represents the main context used for profiling.
struct GlobalProfilerContext {
  GlobalProfilerContext(DurationType granularity, StringRef name,
                        uint64_t level);

  /// Collect all the begin, end, and sample events over all threads, reconcile
  /// them, and return them as timing entries sorted by time then thread id.
  std::vector<CompletedEntry> getCompletedEntries();

  /// Sets the cpuDevice profiling mask in order to toggle trace types enabled
  /// with MODULAR_ASYNCRT_MAX_PROFILING_LEVEL.
  /// Pre-condition: there must be no profiler events currently in progress.
  void setRuntimeProfilingTypeMask(uint64_t typeMask);

  /// Write all the completed entries in JSON form to os, using format in:
  /// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
  void writeJsonTrace(llvm::raw_pwrite_stream &os,
                      ArrayRef<CompletedEntry> entries);

  /// Write all the completed entries in plain text form to os.
  void writeTextTrace(llvm::raw_pwrite_stream &os,
                      ArrayRef<CompletedEntry> entries);

  /// The minimum time granularity for time trace profiler.
  DurationType granularity;

  /// The name of the process this profiler is running on.
  StringRef procName;

  /// The id of the process this profiler is running on.
  const llvm::sys::Process::Pid pid;

  /// System clock time when the session was begun.
  std::chrono::time_point<std::chrono::system_clock> beginningOfTime;

  /// Profiling clock time when the session was begun.
  const TimePointType startTime;

  /// Lock used to guard access to the running profilers.
  std::mutex lock;

  /// The set of running profilers for each thread.
  std::vector<std::unique_ptr<TimeTraceThreadProfiler>> profilers;

  /// The next available thread index, to ensure all ProfilerEventIds are
  /// globally unique across all thread profilers.
  uint16_t nextThreadIndex = 0;

  /// A set of active thread profiler contexts.
  llvm::DenseSet<ThreadProfilerContext *> threadProfilerContexts;

  SmallVector<std::string> inputShapes;

  /// Runtime configurable filter for profiling types (`Trace::Type`).
  /// See TimeTraceThreadProfiler::runtimeProfilingTypeMask for detailed docs.
  uint64_t runtimeProfilingTypeMask;
};

//===----------------------------------------------------------------------===//
// ProfilingDetail::DebugStream
//===----------------------------------------------------------------------===//

struct DebugStream {
  DebugStream() = default;
  ~DebugStream() {
    if (auto *ctx = ThreadProfilerContext::get())
      ctx->debug(std::move(str));
    else
      DebugEvent(0, std::move(str)).dump();
  }

  template <typename T>
  llvm::raw_ostream &operator<<(const T &t) {
    return os << t;
  }

private:
  std::string str;
  llvm::raw_string_ostream os{str};
};

//===----------------------------------------------------------------------===//
// ProfilingDetail::DummyStream
//===----------------------------------------------------------------------===//

struct DummyStream {
  DummyStream() = default;
  ~DummyStream() = default;

  template <typename T>
  llvm::raw_ostream &operator<<(const T &t) {
    return os;
  }

private:
  llvm::raw_null_ostream os;
};

} // namespace ProfilingDetail

//===----------------------------------------------------------------------===//
// TimeTraceProfiler
//===----------------------------------------------------------------------===//

/// This class represents the main time trace profiler, of which only one should
/// ever be active at a given time.
struct TimeTraceProfiler {
  /// Initialize the time trace profiler. This should be constructed from the
  /// main thread.
  /// `runtimeProfilingTypeMask` defaults to fully enabled, but can be set at
  /// cpuDevice to toggle trace types enabled with
  /// MODULAR_ASYNCRT_MAX_PROFILING_LEVEL.
  TimeTraceProfiler(unsigned timeTraceGranularity, StringRef procName,
                    StringRef filename = "",
                    uint64_t runtimeProfilingTypeMask = Trace::kFullyEnabled);

  /// Destroy the time trace profiler. This should be destroyed from the
  /// main thread.
  ~TimeTraceProfiler();

  /// Append given input shape to internal list.
  /// These will be included in metadata written to output stream.
  void addInputShape(const std::string &shape);

  /// Sets the cpuDevice profiling mask in order to toggle trace types enabled
  /// with MODULAR_ASYNCRT_MAX_PROFILING_LEVEL.
  /// Pre-condition: there must be no profiler events currently in progress.
  void setRuntimeProfilingTypeMask(uint64_t typeMask);

  /// Get the cpuDevice profiling mask that toggles trace types enabled with
  /// MODULAR_ASYNCRT_MAX_PROFILING_LEVEL.
  uint64_t runtimeProfilingTypeMask();

  /// Write profiling data to a file.
  /// The function will write to `profileFilename` if present, if not then will
  /// write to fallbackFileName appending .time-trace. Returns a StringError
  /// indicating a failure if the function is unable to open the file for
  /// writing.
  ErrorOrSuccess write(StringRef fallbackFileName);

  /// Writes the profiling data in JSON form to os. Visible for testing.
  void writeJSONForTesting(llvm::raw_pwrite_stream &os);

  /// Make sure all internable strings are captured in all profiling entries.
  void intern();

  /// Filename into which time profiling should be written, or the empty
  /// string if disabled.
  std::string profileFilename;
};

//===----------------------------------------------------------------------===//
// ProfilerEntry
//===----------------------------------------------------------------------===//

///
/// Represents an open or completed timing/sampling tracing entry. However
/// if Enabled is false, will be the trivial empty struct. Timing entries
/// capture the beginning and end timestamps for a named event. Sampling
/// entries capture a single size_t value sampling a named value of interest.
/// The duration of sampling entries is ignored by viewer, however sampling
/// entries must still be recorded by invoking 'record()'.
///
/// Here's the interface supported for entries:
///   -- Empty entry, never recorded.
///   ProfilerEntry()
///
///   -- Start recording a timing entry with name and result of detailFn.
///   -- CAUTION: Both must be literals to guarantee zero-cost when
///   -- profiling disabled at compile time.
///   static ProfilerEntry
///   create(StringRef name, llvm::function_ref<std::string()> detailFn);
///
///   -- Ditto, but detail is literal string.
///   -- CAUTION: Both must be literals to guarantee zero-cost when
///   -- profiling disabled at compile time.
///   static ProfilerEntry create(StringRef name, StringRef detail);
///
///   -- Start recording a sampling entry with name and result of valueFn.
///   -- CAUTION: Both must be literals to guarantee zero-cost when
///   -- profiling disabled at compile time.
///   static ProfilerEntry
///   create(StringRef name, llvm::function_ref<size_t()> valueFn);
///
///   -- Ditto, but value is already computed.
///   -- CAUTION: Both must be literals to guarantee zero-cost when
///   -- profiling disabled at compile time.
///   static ProfilerEntry create(StringRef name, size_t value);
///
///   -- Return true if entry is empty.
///   bool empty() const;
///
///   -- Stop the entry's clock, and move the entry into the profiling
///   -- database.
///   void record() &&
///
///   -- Ditto, but pass a custom function that returns an arbitrary
///   -- `std::string`.
///   -- {Begin,End}Event dump returned strings to `llvm::dbgs()`.
///   void record(ProfilerDumpFn dumpFn) &&
template <bool Enabled, Trace::Type Type>
struct ProfilerEntry {};

/// Disabled profiling entry. Everything is a no-op.
template <Trace::Type Type>
struct ProfilerEntry<false, Type> {
  // No copy, only move.
  ProfilerEntry(const ProfilerEntry &) = delete;
  ProfilerEntry &operator=(const ProfilerEntry &) = delete;
  ProfilerEntry(ProfilerEntry &&) = default;
  ProfilerEntry &operator=(ProfilerEntry &&) = default;

  ProfilerEntry() = default;

  static constexpr bool isEnabled() { return false; }

  template <typename... Args>
  static ProfilerEntry create(Args &&...args) {
    return {};
  }

  template <typename... Args>
  static ProfilerEntry createWithParent(ProfilerEventId parentId,
                                        Args &&...args) {
    return {};
  }

  template <typename... Args>
  static void createAndPush(Args &&...args) {}

  static void endAndPop() {}

  static ProfilerEventId getCurrentId() { return 0; }
  void setAsCurrentId() {}
  static void clearCurrentId() {}

  template <typename... Args>
  static void sample(uint64_t value, Args &&...args) {}

  static ProfilingDetail::DummyStream debug() {
    return ProfilingDetail::DummyStream();
  }

  bool empty() const { return true; }
  ProfilerEventId getId() const { return 0; }

  template <typename... Args>
  void record(Args &&...args) && {}
};

/// Enabled profiling entry. Entries are created only if the profiler is active.
template <Trace::Type Type>
struct ProfilerEntry<true, Type> {
  ProfilerEntry(ProfilerEventId id) : id(id) {}

  // No copy, only move.
  ProfilerEntry(const ProfilerEntry &) = delete;
  ProfilerEntry &operator=(const ProfilerEntry &) = delete;
  ProfilerEntry(ProfilerEntry &&) = default;
  ProfilerEntry &operator=(ProfilerEntry &&) = default;

  ProfilerEntry() = default;

  static constexpr bool isEnabled() { return true; }

  template <typename... Args>
  static ProfilerEntry create(Args &&...args) {
    if (auto *ctx = ProfilingDetail::ThreadProfilerContext::get()) {
      if (!ctx->isEnabled(Type)) {
        // Skip recording events that are turned off at runtime.
        return {};
      }

      return ProfilerEntry(ctx->begin(std::forward<Args>(args)...));
    }
    return {};
  }

  template <size_t N, typename... Args>
  static ProfilerEntry create(const char (&s)[N], Args &&...args) {
    if (auto *ctx = ProfilingDetail::ThreadProfilerContext::get()) {
      if (!ctx->isEnabled(Type))
        return {};

      return ProfilerEntry(ctx->begin(StringLiteral::withInnerNUL(s),
                                      std::forward<Args>(args)...));
    }
    return {};
  }

  template <typename... Args>
  static ProfilerEntry createWithParent(ProfilerEventId parentId,
                                        Args &&...args) {
    if (auto *ctx = ProfilingDetail::ThreadProfilerContext::get()) {
      if (!ctx->isEnabled(Type))
        return {};

      return ProfilerEntry(
          ctx->beginWithParent(parentId, std::forward<Args>(args)...));
    }

    return {};
  }

  template <typename... Args>
  static void createAndPush(Args &&...args) {
    if (auto *ctx = ProfilingDetail::ThreadProfilerContext::get()) {
      if (!ctx->isEnabled(Type))
        return;

      ctx->beginAndPush(std::forward<Args>(args)...);
    }
  }

  template <typename... Args>
  static void createWithParentAndPush(ProfilerEventId parentId,
                                      Args &&...args) {
    if (auto *ctx = ProfilingDetail::ThreadProfilerContext::get()) {
      if (!ctx->isEnabled(Type))
        return;

      ctx->beginWithParentAndPush(parentId, std::forward<Args>(args)...);
    }
  }

  template <size_t N, typename... Args>
  static void createAndPush(const char (&s)[N], Args &&...args) {
    if (auto *ctx = ProfilingDetail::ThreadProfilerContext::get()) {
      if (!ctx->isEnabled(Type))
        return;

      ctx->beginAndPush(StringLiteral::withInnerNUL(s),
                        std::forward<Args>(args)...);
    }
  }

  static void endAndPop() {
    if (auto *ctx = ProfilingDetail::ThreadProfilerContext::get()) {
      if (!ctx->isEnabled(Type))
        return;

      ctx->endAndPop();
    }
  }

  /// Returns the event id of the 'current' event for the callers thread.
  /// May be 0 to indicate no parent.
  ///
  /// This is a convenience to convey the notion of 'parent' event into
  /// child events created by a subroutine. However, since events may be
  /// begun on one thread and finished on another, care must be taken to
  /// keep this notion of 'parent' event accurate. See clearCurrentId().
  static ProfilerEventId getCurrentId() {
    if (auto *ctx = ProfilingDetail::ThreadProfilerContext::get())
      return ctx->getCurrentId();
    return 0;
  }

  /// Records this profiling entry as the 'current' event for the caller's
  /// thread. See also clearCurrentId().
  void setAsCurrentId() {
    if (auto *ctx = ProfilingDetail::ThreadProfilerContext::get())
      ctx->setCurrentId(id);
  }

  /// Clears the 'current' event for the caller's thread. This should be called
  /// before execution returns out of the logical scope of a parent profiling
  /// entry.
  static void clearCurrentId() {
    if (auto *ctx = ProfilingDetail::ThreadProfilerContext::get())
      ctx->setCurrentId(0);
  }

  template <typename... Args>
  static void sample(uint64_t value, Args &&...args) {
    if (auto *ctx = ProfilingDetail::ThreadProfilerContext::get())
      ctx->sample(value, std::forward<Args>(args)...);
  }

  static ProfilingDetail::DebugStream debug() {
    return ProfilingDetail::DebugStream();
  }

  bool empty() const { return id == 0; }
  ProfilerEventId getId() const { return id; }

  template <typename... Args>
  void record(Args &&...args) && {
    if (id == 0)
      return;
    if (auto *ctx = ProfilingDetail::ThreadProfilerContext::get())
      return ctx->end(id, std::forward<Args>(args)...);
  }

private:
  ProfilerEventId id = 0;
};

//===----------------------------------------------------------------------===//
// TimeTraceScope
//===----------------------------------------------------------------------===//

struct Empty {};

/// RAII class to automatically record the constructed or given profile entry
/// when the object goes out of scope.
template <Trace::Type Type, bool Enabled = true, typename DumpFnT = Empty>
struct TimeTraceScope {
  static_assert(std::is_same_v<DumpFnT, Empty> ||
                    std::is_constructible_v<ProfilerDumpFn, DumpFnT>,
                "DumpFnT must be either Empty or ProfilerDumpFn.");

  TimeTraceScope() = delete;
  TimeTraceScope(const TimeTraceScope &) = delete;
  TimeTraceScope &operator=(const TimeTraceScope &) = delete;
  TimeTraceScope(TimeTraceScope &&) = delete;
  TimeTraceScope &operator=(TimeTraceScope &&) = delete;

  explicit TimeTraceScope(ProfilerEntry<Enabled, Type> entry,
                          DumpFnT extraDumpFn = Empty{})
      : entry(std::move(entry)), dumpFn(std::move(extraDumpFn)) {}

  ~TimeTraceScope() {
    if constexpr (std::is_same_v<DumpFnT, Empty>)
      std::move(entry).record();
    else
      std::move(entry).record(dumpFn);
  }

  ProfilerEntry<Enabled, Type> entry;
  /// Optional function to dump statistics upon profiler events occurring.
  /// When present, this is a `ProfilerDumpFn`.
  /// Otherwise, this is an empty struct `Empty` so as to be cost free.
  /// Since few profiler entries will actually want a `dumpFn` in practice, this
  /// is enabled/disabled separately from the `Enabled` template argument.
  DumpFnT dumpFn;
};

// The trivial deduction guide.
template <Trace::Type Type, bool Enabled>
TimeTraceScope(ProfilerEntry<Enabled, Type> &&)
    -> TimeTraceScope<Type, Enabled>;

} // namespace M

#endif // SUPPORT_PROFILING_TIMEPROFILER_H
