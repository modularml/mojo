# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from std.collections import Array
from std.atomic import Atomic, Ordering
from std.sys.intrinsics import (
    ballot,
    implicitarg_ptr,
    llvm_intrinsic,
    readfirstlane,
    sendmsg,
)

from std._gpu.primitives.id import lane_id
from std.collections import Span
from std.memory.pointer import _Null
from std.os import abort

# NOTE: MOST OF THE CODE HERE IS ADAPTED FROM
# AMD'S `device-libs`.
# It is important that the ABI matches up!
# https://github.com/ROCm/llvm-project/tree/656552edc693e2bb4abc9258399c39d190fce2b3/amd/device-libs

# ===-----------------------------------------------------------------------===#
# HSA Queue Ops
# ===-----------------------------------------------------------------------===#

# Matches the ABI of:
# https://github.com/ROCm/llvm-project/blob/656552edc693e2bb4abc9258399c39d190fce2b3/amd/device-libs/ockl/inc/amd_hsa_signal.h#L50
comptime amd_signal_kind64_t = Int64


# Must match the ABI of:
# https://github.com/ROCm/llvm-project/blob/656552edc693e2bb4abc9258399c39d190fce2b3/amd/device-libs/ockl/inc/amd_hsa_signal.h#L61
@fieldwise_init
struct amd_signal_t(Copyable):
    var kind: amd_signal_kind64_t
    var value: UInt64
    var event_mailbox_ptr: UInt64
    var event_id: UInt32
    var reserved1: UInt32
    var start_ts: UInt64
    var end_ts: UInt64
    var reserved2: UInt64
    var reserved3: Array[UInt32, 2]


@always_inline
def update_mbox(sig: ImmPointer[amd_signal_t, ...]):
    var mb = sig[].event_mailbox_ptr
    if Int(mb) != Int(_Null[address_space=.GLOBAL]()):
        var mb_ptr = Pointer[
            UInt64, UntrackedOrigin[mut=True], address_space=.GLOBAL
        ](unsafe_from_address=Int(mb))
        var id = sig[].event_id.cast[.uint64]()
        Atomic.store[ordering=Ordering.RELEASE](mb_ptr, id)
        sendmsg(1 | (0 << 4), readfirstlane(id.cast[.int32]()) & 0xFF)


@always_inline
def hsa_signal_add(sig: UInt64, value: UInt64):
    var s = Pointer(to=sig).unsafe_bitcast[
        Pointer[amd_signal_t, MutUntrackedOrigin, address_space=.GLOBAL]
    ]()[]
    _ = Atomic.fetch_add[ordering=Ordering.RELEASE](
        Pointer(to=s[].value), value
    )
    update_mbox(s)


# ===-----------------------------------------------------------------------===#
# Services
# ===-----------------------------------------------------------------------===#


# Matches the values described in:
# https://github.com/ROCm/llvm-project/blob/656552edc693e2bb4abc9258399c39d190fce2b3/amd/device-libs/ockl/src/services.cl#L21
struct ServiceId:
    comptime reserved = 0
    comptime function_call = 1
    comptime printf = 2
    comptime fprintf = Self.printf
    comptime devmem = 3
    comptime sanitizer = 4


# Matches the values described in:
# https://github.com/ROCm/llvm-project/blob/656552edc693e2bb4abc9258399c39d190fce2b3/amd/device-libs/ockl/src/services.cl#L87
struct DescriptorOffset:
    comptime flag_begin = 0
    comptime flag_end = 1
    comptime reserved0 = 2
    comptime len = 5
    comptime id = 8


# Matches the values described in:
# https://github.com/ROCm/llvm-project/blob/656552edc693e2bb4abc9258399c39d190fce2b3/amd/device-libs/ockl/src/services.cl#L95
struct DescriptorWidth:
    comptime flag_begin = 1
    comptime flag_end = 1
    comptime reserved0 = 3
    comptime len = 3
    comptime id = 56


@always_inline
def msg_set_len(pd: UInt64, len: UInt32) -> UInt64:
    var reset_mask = ~(
        ((UInt64(1) << DescriptorWidth.len) - 1) << DescriptorOffset.len
    )
    return (pd & reset_mask) | (len.cast[.uint64]() << DescriptorOffset.len)


@always_inline
def msg_set_begin_flag(pd: UInt64) -> UInt64:
    return pd | (UInt64(1) << DescriptorOffset.flag_begin)


@always_inline
def msg_reset_begin_flag(pd: UInt64) -> UInt64:
    return pd & (~(UInt64(1) << DescriptorOffset.flag_begin))


@always_inline
def msg_get_end_flag(pd: UInt64) -> UInt64:
    return pd & (UInt64(1) << DescriptorOffset.flag_end)


@always_inline
def msg_reset_end_flag(pd: UInt64) -> UInt64:
    return pd & (~(UInt64(1) << DescriptorOffset.flag_end))


@always_inline
def msg_set_end_flag(pd: UInt64) -> UInt64:
    return pd | (UInt64(1) << DescriptorOffset.flag_end)


def append_bytes(
    service_id: UInt32,
    msg_desc: UInt64,
    mut data: Span[UInt8, _],
) -> Tuple[UInt64, UInt64]:
    var msg_desc_ = msg_set_len(msg_desc, UInt32((len(data) + 7) // 8))

    @always_inline
    def pack_uint64() {mut} -> UInt64:
        var arg = UInt64(0)
        if len(data) >= 8:
            arg = data.unsafe_ptr().unsafe_bitcast[UInt64]()[]
            data = data[8:]
        else:
            var ii = 0
            for byte in data:
                arg |= byte.cast[.uint64]() << UInt64(ii * 8)
                ii += 1
            data = data[0:0]
        return arg

    var arg1 = pack_uint64()
    var arg2 = pack_uint64()
    var arg3 = pack_uint64()
    var arg4 = pack_uint64()
    var arg5 = pack_uint64()
    var arg6 = pack_uint64()
    var arg7 = pack_uint64()
    return hostcall(
        service_id,
        msg_desc_,
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        arg7,
    )


@no_inline
def message_append_bytes(
    service_id: UInt32, msg_desc: UInt64, data: Span[UInt8, _]
) -> Tuple[UInt64, UInt64]:
    """
    Append an array of bytes to a message.

    Args:
        service_id: Identifier for the target host-side service.
        msg_desc: Message descriptor for a new or existing message.
        data: Span with an array of bytes.

    Returns:
        Values depend on the state of the message.

    The function can transmit a byte array of arbitrary length, but
    during transmission, the array is padded with zeroes until the
    length is a multiple of eight bytes. Only the array contents are
    transmitted, and not the length.

    If the END flag is set, the function returns two long values
    received from the host message handler. Otherwise, the first
    return value is the message descriptor to be used for a subsequent
    message call, while the second return value is not defined.

    """
    var data_ = data
    var end_flag = msg_get_end_flag(msg_desc)
    var retval = (UInt64(0), UInt64(0))
    retval[0] = msg_reset_end_flag(msg_desc)

    while True:
        var prev_len = len(data_)
        # We can only send 7 packed UInt64s per message
        # Therefore, if the length is greater than 56,
        # we need to take a 56 byte == (7 * size_of[UInt64]())
        # chunk to process.
        if len(data_) > 56:
            prev_len = 56
        else:
            retval[0] |= end_flag
        var d = data_[:prev_len]
        retval = append_bytes(service_id, retval[0], d)
        data_ = data_[prev_len:]
        if not data_:
            break

    return retval


@always_inline
def message_append_args(
    service_id: UInt32,
    msg_desc: UInt64,
    num_args: UInt32,
    arg0: UInt64,
    arg1: UInt64,
    arg2: UInt64,
    arg3: UInt64,
    arg4: UInt64,
    arg5: UInt64,
    arg6: UInt64,
) -> Tuple[UInt64, UInt64]:
    """
    Append up to seven ulong values to a message.

    Args:
        service_id: Identifier for the target host-side service.
        msg_desc: Message descriptor for a new or existing message.
        num_args: Number of arguments to be appended (maximum seven).
        arg0: Argument to be appended.
        arg1: Argument to be appended.
        arg2: Argument to be appended.
        arg3: Argument to be appended.
        arg4: Argument to be appended.
        arg5: Argument to be appended.
        arg6: Argument to be appended.

    Returns:
        Values depend on the state of the message.

    Only the first num_args arguments are appended to the
    message. The remaining arguments are ignored. Behaviour is
    undefined if num_args is greater then seven.

    If the END flag is set, the function returns two uint64_t values
    received from the host message handler. Otherwise, the first
    return value is the message descriptor to be used for a subsequent
    message call, while the second return value is not defined.
    """
    var msg_desc_ = msg_set_len(msg_desc, num_args)

    return hostcall(
        service_id,
        msg_desc_,
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
    )


# ===-----------------------------------------------------------------------===#
# Services - printf
# ===-----------------------------------------------------------------------===#


# Matches the values described in:
# https://github.com/ROCm/llvm-project/blob/656552edc693e2bb4abc9258399c39d190fce2b3/amd/device-libs/ockl/src/services.cl#L243
struct FprintfCtrl:
    comptime stdout = 0
    comptime stderr = 1


@always_inline
def begin_fprintf(flags: UInt32) -> UInt64:
    # The two standard output streams stderr and stdout are indicated
    # using the lowest bits in the control qword. For now, all other
    # bits are required to be zero.
    var msg_desc = msg_set_begin_flag(0)
    var control = flags.cast[.uint64]()

    var retval = message_append_args(
        ServiceId.fprintf,
        msg_desc,
        1,  # num args
        control,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    return retval[0]


@always_inline
def fprintf_stdout_begin() -> UInt64:
    """
    Begin a new fprintf message for stdout.
    Returns:
        Message descriptor for a new printf invocation.
    """
    return begin_fprintf(FprintfCtrl.stdout)


@always_inline
def fprintf_stderr_begin() -> UInt64:
    """
    Begin a new fprintf message for stderr.

    Returns:
        Message descriptor for a new printf invocation.
    """
    return begin_fprintf(FprintfCtrl.stderr)


@always_inline
def fprintf_append_args(
    msg_desc: UInt64,
    num_args: UInt32,
    value0: UInt64,
    value1: UInt64,
    value2: UInt64,
    value3: UInt64,
    value4: UInt64,
    value5: UInt64,
    value6: UInt64,
    is_last: Int32,
) -> UInt64:
    """
    Append up to seven arguments to the fprintf message.

    Args:
        msg_desc:  Message descriptor for the current fprintf.
        num_args:  Number of arguments to be appended (maximum seven).
        value0: The argument values to be appended.
        value1: The argument values to be appended.
        value2: The argument values to be appended.
        value3: The argument values to be appended.
        value4: The argument values to be appended.
        value5: The argument values to be appended.
        value6: The argument values to be appended.
        is_last:   If non-zero, this causes the fprintf to be completed.

    Returns:
        Value depends on is_last.

    Only the first num_args arguments are appended to the
    message. The remaining arguments are ignored. Behaviour is
    undefined if num_args is greater then seven.

    If is_last is zero, the function returns a message descriptor that
    must be used by a subsequent call to any __ockl_fprintf*
    function. If is_last is non-zero, the function causes the current
    fprintf to be completed on the host-side, and returns the value
    returned by that fprintf.

    """
    var msg_desc_ = msg_desc
    if is_last:
        msg_desc_ = msg_set_end_flag(msg_desc_)

    var retval = message_append_args(
        ServiceId.fprintf,
        msg_desc_,
        num_args,
        value0,
        value1,
        value2,
        value3,
        value4,
        value5,
        value6,
    )
    return retval[0]


@always_inline
def fprintf_append_string_n(
    msg_desc: UInt64, data: Span[UInt8, _], is_last: Bool
) -> UInt64:
    """
    Append a null-terminated string to the fprintf message.

    Args:
        msg_desc: Message descriptor for the current fprintf.
        data: Span with the bytes of the string, including null terminator.
        is_last: If non-zero, this causes the fprintf to be completed.

    Returns:
        Value depends on is_last.

     The function appends a single null-terminated string to a current
     fprintf message, including the final null character. The host-side
     can use the bytes as a null-terminated string in place, without
     having to first copy the string and then append the null
     terminator.

     length itself is not transmitted. Behaviour is undefined if
     length does not include the final null character. data may
     be a null pointer, in which case, length is ignored and a single
     zero is transmitted. This makes the nullptr indistinguishable from
     an empty string to the host-side receiver.

     The call to message_append_args() ensures that during
     transmission, the string is null-padded to a multiple of eight.

     If is_last is zero, the function returns a message descriptor that
     must be used by a subsequent call to any __ockl_fprintf*
     function. If is_last is non-zero, the function causes the current
     fprintf to be completed on the host-side, and returns the value
     returned by that fprintf.

    """
    var msg_desc_ = msg_desc

    if is_last:
        msg_desc_ = msg_set_end_flag(msg_desc_)

    if not data:
        var retval = message_append_args(
            ServiceId.fprintf,
            msg_desc_,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )

        return retval[0]
    var retval = message_append_bytes(ServiceId.fprintf, msg_desc_, data)
    return retval[0]


@always_inline
def printf_begin() -> UInt64:
    return fprintf_stdout_begin()


@always_inline
def printf_append_args(
    msg_desc: UInt64,
    num_args: UInt32,
    value0: UInt64 = 0,
    value1: UInt64 = 0,
    value2: UInt64 = 0,
    value3: UInt64 = 0,
    value4: UInt64 = 0,
    value5: UInt64 = 0,
    value6: UInt64 = 0,
    is_last: Int32 = 0,
) -> UInt64:
    return fprintf_append_args(
        msg_desc,
        num_args,
        value0,
        value1,
        value2,
        value3,
        value4,
        value5,
        value6,
        is_last,
    )


@always_inline
def printf_append_string_n(
    msg_desc: UInt64, data: Span[UInt8, _], is_last: Bool
) -> UInt64:
    return fprintf_append_string_n(msg_desc, data, is_last)


# ===-----------------------------------------------------------------------===#
# Hostcall
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct Header(TrivialRegisterPassable):
    var _handle: Pointer[header_t, MutUntrackedOrigin, address_space=.GLOBAL]

    def fill_packet(
        mut self,
        mut payload: Payload,
        service_id: UInt32,
        arg0: UInt64,
        arg1: UInt64,
        arg2: UInt64,
        arg3: UInt64,
        arg4: UInt64,
        arg5: UInt64,
        arg6: UInt64,
        arg7: UInt64,
        me: UInt32,
        low: UInt32,
    ):
        var active = ballot[.int64](True).cast[.uint64]()
        if me == low:
            var control = set_ready_flag(0)
            self._handle[].control = control
            self._handle[].activemask = active
            self._handle[].service = service_id

        payload[Int(me), 0] = arg0
        payload[Int(me), 1] = arg1
        payload[Int(me), 2] = arg2
        payload[Int(me), 3] = arg3
        payload[Int(me), 4] = arg4
        payload[Int(me), 5] = arg5
        payload[Int(me), 6] = arg6
        payload[Int(me), 7] = arg7

    def get_return_value(
        mut self, payload: Payload, me: UInt32, low: UInt32
    ) -> Tuple[UInt64, UInt64]:
        """
        Wait for the host response and return the first two ulong
        entries per workitem.

        After the packet is submitted in READY state, the wave spins until
        the host changes the state to DONE. Each workitem reads the first
        two ulong elements in its slot and returns this.
        """
        # The while loop needs to be executed by all active
        # lanes. Otherwise, later reads from ptr are performed only by
        # the first thread, while other threads reuse a value cached from
        # previous operations. The use of readfirstlane in the while loop
        # prevents this reordering.
        #
        # In the absence of the readfirstlane, only one thread has a
        # sequenced-before relation from the atomic load on
        # header->control to the ordinary loads on ptr. As a result, the
        # compiler is free to reorder operations in such a way that the
        # ordinary loads are performed only by the first thread. The use
        # of readfirstlane provides a stronger code-motion barrier, and
        # it effectively "spreads out" the sequenced-before relation to
        # the ordinary stores in other threads too.
        while True:
            var ready_flag = UInt32(1)
            if me == low:
                var ptr = Pointer(to=self._handle[].control)
                var control = Atomic.load[ordering=Ordering.ACQUIRE](ptr)
                ready_flag = get_ready_flag(control)

            ready_flag = readfirstlane(ready_flag)

            if ready_flag == 0:
                break

            llvm_intrinsic["llvm.amdgcn.s.sleep", NoneType](Int32(1))

        ref ptr = payload._handle[].slots[Int(me)]
        var value0 = ptr[0]
        var value1 = ptr[1]
        return value0, value1


# Must match the ABI of:
# https://github.com/ROCm/llvm-project/blob/656552edc693e2bb4abc9258399c39d190fce2b3/amd/device-libs/ockl/src/hostcall_impl.cl#L30
# but this is actually just conforming to the ABI of:
# https://github.com/ROCm/clr/blob/f5b2516f5d8a44b06ad1907594db1be25a9fe57b/rocclr/device/devhostcall.hpp#L104
@fieldwise_init
struct header_t(TrivialRegisterPassable):
    var next: UInt64
    var activemask: UInt64
    var service: UInt32
    var control: UInt32


@fieldwise_init
struct Payload(TrivialRegisterPassable):
    var _handle: Pointer[payload_t, MutUntrackedOrigin]

    def __getitem__(self, idx0: Int, idx1: Int) -> UInt64:
        abort("shouldn't load from this")

    @always_inline
    def __setitem__(mut self, idx0: Int, idx1: Int, value: UInt64):
        self._handle[].slots[idx0][idx1] = value


# Must match the ABI of:
# https://github.com/ROCm/llvm-project/blob/656552edc693e2bb4abc9258399c39d190fce2b3/amd/device-libs/ockl/src/hostcall_impl.cl#L37
# but this is actually just conforming to the ABI of:
# https://github.com/ROCm/clr/blob/f5b2516f5d8a44b06ad1907594db1be25a9fe57b/rocclr/device/devhostcall.hpp#L99
@fieldwise_init
struct payload_t(Copyable):
    var slots: Array[Array[UInt64, 8], 64]


@fieldwise_init
struct Buffer(TrivialRegisterPassable):
    var _handle: Pointer[buffer_t, MutUntrackedOrigin, address_space=.GLOBAL]

    @always_inline
    def get_header(self, ptr: UInt64) -> Header:
        return Header(
            self._handle[].headers.unsafe_offset(
                ptr & self._handle[].index_mask
            )
        )

    @always_inline
    def get_payload(self, ptr: UInt64) -> Payload:
        return Payload(
            self._handle[].payloads.unsafe_offset(
                ptr & self._handle[].index_mask
            )
        )

    def pop(mut self, top: MutPointer[UInt64, ...]) -> UInt64:
        var f = Atomic.load[ordering=Ordering.ACQUIRE](top)
        # F is guaranteed to be non-zero, since there are at least as
        # many packets as there are waves, and each wave can hold at most
        # one packet.
        while True:
            var p = self.get_header(f)
            var n = Atomic.load[ordering=Ordering.RELAXED](
                Pointer(to=p._handle[].next)
            )
            if Atomic.compare_exchange[
                success_ordering=Ordering.ACQUIRE,
                failure_ordering=Ordering.RELAXED,
            ](top, f, n):
                break
            llvm_intrinsic["llvm.amdgcn.s.sleep", NoneType](Int32(1))
        return f

    def pop_free_stack(mut self, me: UInt32, low: UInt32) -> UInt64:
        """
        Use the first active lane to get a free packet and
        broadcast to the whole wave.
        """
        var packet_ptr = UInt64(0)
        if me == low:
            packet_ptr = Self.pop(
                self,
                Pointer(to=self._handle[].free_stack),
            )

        return readfirstlane(packet_ptr)

    def push(mut self, top: MutPointer[UInt64, ...], ptr: UInt64):
        var f = Atomic.load[ordering=Ordering.RELAXED](top)
        var p = self.get_header(ptr)
        while True:
            p._handle[].next = f
            if Atomic.compare_exchange[
                success_ordering=Ordering.RELEASE,
                failure_ordering=Ordering.RELAXED,
            ](top, f, ptr):
                break
            llvm_intrinsic["llvm.amdgcn.s.sleep", NoneType](Int32(1))

    def push_ready_stack(mut self, ptr: UInt64, me: UInt32, low: UInt32):
        """
        Use the first active lane in a wave to submit a ready
        packet and signal the host.
        """
        if me == low:
            self.push(Pointer(to=self._handle[].ready_stack), ptr)
            send_signal(self._handle[].doorbell)

    def return_free_packet(mut self, ptr: UInt64, me: UInt32, low: UInt32):
        """
        Return the packet after incrementing the ABA tag.
        """
        if me == low:
            var ptr = inc_ptr_tag(ptr, self._handle[].index_mask)
            self.push(Pointer(to=self._handle[].free_stack), ptr)


# Must match the ABI of:
# https://github.com/ROCm/llvm-project/blob/656552edc693e2bb4abc9258399c39d190fce2b3/amd/device-libs/ockl/src/hostcall_impl.cl#L45
# but this is actually just conforming to the ABI of:
# https://github.com/ROCm/clr/blob/f5b2516f5d8a44b06ad1907594db1be25a9fe57b/rocclr/device/devhostcall.hpp#L144
#
# AMD's note: Hostcall buffer struct defined here is not an exact
# match of runtime buffer layout but matches its prefix that
# this code tries to access.
@fieldwise_init
struct buffer_t(Copyable, TrivialRegisterPassable):
    var headers: Pointer[header_t, MutUntrackedOrigin, address_space=.GLOBAL]
    var payloads: Pointer[payload_t, MutUntrackedOrigin]
    var doorbell: UInt64
    var free_stack: UInt64
    var ready_stack: UInt64
    var index_mask: UInt64


@fieldwise_init
struct ControlOffset(TrivialRegisterPassable):
    var value: UInt32
    comptime ready_flag = Self(0)
    comptime reserved0 = Self(1)

    @always_inline
    def __ne__(self, rhs: Self) -> Bool:
        return self.value != rhs.value

    @always_inline
    def __eq__(self, rhs: Self) -> Bool:
        return self.value == rhs.value


@fieldwise_init
struct ControlWidth(TrivialRegisterPassable):
    var value: UInt32
    comptime ready_flag = Self(1)
    comptime reserved0 = Self(31)

    @always_inline
    def __ne__(self, rhs: Self) -> Bool:
        return self.value != rhs.value

    @always_inline
    def __eq__(self, rhs: Self) -> Bool:
        return self.value == rhs.value


@always_inline
def get_control_mask(control: UInt32, offset: UInt32, width: UInt32) -> UInt32:
    var value: UInt32 = (control >> offset) & ((UInt32(1) << width) - UInt32(1))
    return value


@always_inline
def get_control_field(
    control: UInt32, offset: ControlOffset, width: ControlWidth
) -> UInt32:
    var value: UInt32 = (control >> offset.value) & (
        (UInt32(1) << width.value) - 1
    )
    return value


@always_inline
def set_control_field(
    control: UInt32, offset: ControlOffset, width: ControlWidth, value: UInt32
) -> UInt32:
    var mask: UInt32 = ~(((UInt32(1) << width.value) - 1) << offset.value)
    return (control & mask) | (value << offset.value)


@always_inline
def get_ready_flag(control: UInt32) -> UInt32:
    return get_control_field(
        control, ControlOffset.ready_flag, ControlWidth.ready_flag
    )


@always_inline
def set_ready_flag(control: UInt32) -> UInt32:
    return set_control_field(
        control, ControlOffset.ready_flag, ControlWidth.ready_flag, 1
    )


@always_inline
def inc_ptr_tag(ptr: UInt64, index_mask: UInt64) -> UInt64:
    var inc = index_mask + 1
    var ptr_ = ptr + inc
    # Unit step for the tag.
    # When the tag for index 0 wraps, increment the tag.
    if ptr_ == 0:
        return inc
    return ptr_


@always_inline
def send_signal(signal: UInt64):
    hsa_signal_add(signal, 1)


@no_inline
def hostcall(
    service_id: UInt32,
    arg0: UInt64,
    arg1: UInt64,
    arg2: UInt64,
    arg3: UInt64,
    arg4: UInt64,
    arg5: UInt64,
    arg6: UInt64,
    arg7: UInt64,
) -> Tuple[UInt64, UInt64]:
    """
    Submit a wave-wide hostcall packet.

    Args:
        service_id: The service to be invoked on the host.
        arg0: A parameter.
        arg1: A parameter.
        arg2: A parameter.
        arg3: A parameter.
        arg4: A parameter.
        arg5: A parameter.
        arg6: A parameter.
        arg7: A parameter.

    Returns:
        Two 64-bit values.

    The hostcall is executed for all active threads in the
    wave. service_id must be uniform across the active threads,
    otherwise behaviour is undefined. The service parameters may be
    different for each active thread, and correspondingly, the
    returned values are also different.

    The contents of the input parameters and the return values are
    defined by the service being invoked.

    *** PREVIEW FEATURE ***
    This is a feature preview and considered alpha quality only;
    behaviour may vary between ROCm releases. Device code that invokes
    hostcall can be launched only on the ROCm release that it was
    compiled for, otherwise behaviour is undefined.
    """
    var buffer = Buffer(
        implicitarg_ptr().unsafe_bitcast[
            Pointer[buffer_t, MutUntrackedOrigin, address_space=.GLOBAL]
        ]()[unsafe_offset=10]
    )

    var me = UInt32(lane_id())
    var low = readfirstlane(me)

    var packet_ptr = buffer.pop_free_stack(me, low)

    var header = buffer.get_header(packet_ptr)
    var payload = buffer.get_payload(packet_ptr)

    header.fill_packet(
        payload,
        service_id,
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        arg7,
        me,
        low,
    )

    buffer.push_ready_stack(packet_ptr, me, low)
    var retval = header.get_return_value(payload, me, low)
    buffer.return_free_packet(packet_ptr, me, low)
    return retval
