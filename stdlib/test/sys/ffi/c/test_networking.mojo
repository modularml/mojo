# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# RUN: %mojo %s

from testing import assert_equal, assert_false, assert_raises, assert_true
from sys.info import is_big_endian, sizeof
from memory import stack_allocation, UnsafePointer
from utils import StaticTuple
from sys.info import os_is_macos, os_is_linux, os_is_windows

from sys.ffi.c.libc import Libc, TryLibc
from sys.ffi.c.types import (
    C,
    char_ptr,
    in_addr,
    char_ptr_to_string,
    sockaddr_in,
    sockaddr,
    addrinfo,
    socklen_t,
)
from sys.ffi.c.constants import *


def _test_htonl(libc: Libc):
    value = UInt32(1 << 31)
    res = libc.htonl(value)

    @parameter
    if is_big_endian():
        assert_equal(value, res)
    else:
        assert_equal(1 << 7, res)


def test_dynamic_htonl():
    _test_htonl(Libc[static=False]())


def test_static_htonl():
    _test_htonl(Libc[static=True]())


def _test_htons(libc: Libc):
    value = UInt16(1 << 15)
    res = libc.htons(value)

    @parameter
    if is_big_endian():
        assert_equal(value, res)
    else:
        assert_equal(1 << 7, res)


def test_dynamic_htons():
    _test_htons(Libc[static=False]())


def test_static_htons():
    _test_htons(Libc[static=True]())


def _test_ntohl(libc: Libc):
    value = UInt32(1 << 31)
    res = libc.ntohl(value)

    @parameter
    if is_big_endian():
        assert_equal(value, res)
    else:
        assert_equal(1 << 7, res)


def test_dynamic_ntohl():
    _test_ntohl(Libc[static=False]())


def test_static_ntohl():
    _test_ntohl(Libc[static=True]())


def _test_ntohs(libc: Libc):
    value = UInt16(1 << 15)
    res = libc.ntohs(value)

    @parameter
    if is_big_endian():
        assert_equal(value, res)
    else:
        assert_equal(1 << 7, res)


def test_dynamic_ntohs():
    _test_ntohs(Libc[static=False]())


def test_static_ntohs():
    _test_ntohs(Libc[static=True]())


def _test_inet_ntop(libc: Libc):
    ...  # TODO


def test_dynamic_inet_ntop():
    _test_inet_ntop(Libc[static=False]())


def test_static_inet_ntop():
    _test_inet_ntop(Libc[static=True]())


def _test_inet_pton(libc: Libc):
    ...  # TODO


def test_dynamic_inet_pton():
    _test_inet_pton(Libc[static=False]())


def test_static_inet_pton():
    _test_inet_pton(Libc[static=True]())


def _test_inet_addr(libc: Libc):
    ...  # TODO


def test_dynamic_inet_addr():
    _test_inet_addr(Libc[static=False]())


def test_static_inet_addr():
    _test_inet_addr(Libc[static=True]())


def _test_inet_aton(libc: Libc):
    ptr = stack_allocation[1, in_addr]()
    err = libc.inet_aton(char_ptr("123.45.67.89"), ptr)
    assert_true(err != 0)
    res = ptr[0].s_addr
    value = UInt32(0b01111011001011010100001101011001)

    @parameter
    if not is_big_endian():
        b0 = value << 24
        b1 = (value << 8) & 0xFF_00_00
        b2 = (value >> 8) & 0xFF_00
        b3 = value >> 24
        value = b0 | b1 | b2 | b3
    assert_equal(value, res)


def test_dynamic_inet_aton():
    _test_inet_aton(Libc[static=False]())


def test_static_inet_aton():
    _test_inet_aton(Libc[static=True]())


def _test_inet_ntoa(libc: Libc):
    value = UInt32(0b01111011001011010100001101011001)

    @parameter
    if not is_big_endian():
        b0 = value << 24
        b1 = (value << 8) & 0xFF_00_00
        b2 = (value >> 8) & 0xFF_00
        b3 = value >> 24
        value = b0 | b1 | b2 | b3
    res = libc.inet_ntoa(value)
    assert_equal("123.45.67.89", char_ptr_to_string(res))


def test_dynamic_inet_ntoa():
    _test_inet_ntoa(Libc[static=False]())


def test_static_inet_ntoa():
    _test_inet_ntoa(Libc[static=True]())


alias socket_combinations = (
    (AF_INET, SOCK_STREAM, IPPROTO_TCP),
    (AF_INET, SOCK_DGRAM, IPPROTO_UDP),
    (AF_INET, SOCK_SEQPACKET, IPPROTO_SCTP),
    (AF_INET6, SOCK_STREAM, IPPROTO_TCP),
    (AF_INET6, SOCK_DGRAM, IPPROTO_UDP),
    (AF_INET6, SOCK_SEQPACKET, IPPROTO_SCTP),
)


def _test_socket_create(libc: Libc):
    with TryLibc(libc):

        @parameter
        for i in range(len(socket_combinations)):
            alias combo = socket_combinations.get[i, Tuple[Int, Int, Int]]()
            alias address_family = combo.get[0, Int]()
            alias socket_type = combo.get[1, Int]()
            alias socket_protocol = combo.get[2, Int]()

            @parameter
            if os_is_macos():
                if (
                    socket_protocol == IPPROTO_SCTP  # default unsupported
                    or address_family == AF_INET6  # default disabled
                ):
                    continue
            fd = libc.socket(address_family, socket_type, socket_protocol)
            assert_true(fd != -1)
            if socket_protocol == SOCK_STREAM:
                err = libc.shutdown(fd, SHUT_RDWR)
                assert_true(err != -1)


def test_dynamic_socket_create():
    _test_socket_create(Libc[static=False]())


def test_static_socket_create():
    _test_socket_create(Libc[static=True]())


def _test_socketpair(libc: Libc):
    with TryLibc(libc):
        socket_vector = stack_allocation[2, C.int]()
        err = libc.socketpair(AF_LOCAL, SOCK_STREAM, IPPROTO_IP, socket_vector)
        assert_true(err != -1)
        err = libc.shutdown(socket_vector[0], SHUT_RDWR)
        assert_true(err != -1)


def test_dynamic_socketpair():
    _test_socketpair(Libc[static=False]())


def test_static_socketpair():
    _test_socketpair(Libc[static=True]())


def _test_setsockopt(libc: Libc):
    with TryLibc(libc):
        value_ptr = stack_allocation[1, C.int]()
        value_ptr[0] = C.int(1)
        null_ptr = value_ptr.bitcast[C.void]()
        size = socklen_t(sizeof[C.int]())

        @parameter
        if os_is_linux():
            fd = libc.socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)
            assert_true(fd != -1)
            err = libc.setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, null_ptr, size)
            assert_true(err != -1)
            err = libc.setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, null_ptr, size)
            assert_true(err != -1)
            value_ptr[0] = C.int(20)
            err = libc.setsockopt(fd, SOL_TCP, TCP_KEEPIDLE, null_ptr, size)
            assert_true(err != -1)
            err = libc.setsockopt(fd, SOL_TCP, TCP_KEEPINTVL, null_ptr, size)
            assert_true(err != -1)
            err = libc.setsockopt(fd, SOL_TCP, TCP_KEEPCNT, null_ptr, size)
            assert_true(err != -1)
        elif os_is_windows():
            # TODO
            # tcp_keepalive keepaliveParams;
            # DWORD ret = 0;
            # keepaliveParams.onoff = 1;
            # keepaliveParams.keepaliveinterval = keepaliveParams.keepalivetime = keepaliveIntervalSec * 1000;
            # WSAIoctl(sockfd, SIO_KEEPALIVE_VALS, &keepaliveParams, sizeof(keepaliveParams), NULL, 0, &ret, NULL, NULL);
            constrained[False, "Unsupported test"]()
        elif os_is_macos():
            fd = libc.socket(AF_INET, SOCK_STREAM, IPPROTO_IP)
            assert_true(fd != -1)
            err = libc.setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, null_ptr, size)
            assert_true(err != -1)
            err = libc.setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, null_ptr, size)
            assert_true(err != -1)
            value_ptr[0] = C.int(20)
            err = libc.setsockopt(
                fd, IPPROTO_TCP, TCP_KEEPALIVE, null_ptr, size
            )
            assert_true(err != -1)
        else:
            constrained[False, "Unsupported test"]()


def test_dynamic_setsockopt():
    _test_setsockopt(Libc[static=False]())


def test_static_setsockopt():
    _test_setsockopt(Libc[static=True]())


def _test_bind_listen(libc: Libc):
    with TryLibc(libc):
        fd = libc.socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)
        assert_true(fd != -1)
        value_ptr = stack_allocation[1, C.int]()
        value_ptr[0] = 1
        err = libc.setsockopt(
            fd,
            SOL_SOCKET,
            SO_REUSEADDR,
            value_ptr.bitcast[C.void](),
            sizeof[C.int](),
        )
        assert_true(err != -1)
        port = libc.htons(8001)
        ip_buf = stack_allocation[4, C.void]()
        ip_ptr = char_ptr("0.0.0.0")
        err = libc.inet_pton(AF_INET, ip_ptr, ip_buf)
        assert_true(err != 0)
        ip = ip_buf.bitcast[C.u_int]().load()
        zero = StaticTuple[C.char, 8]()
        ai = sockaddr_in(AF_INET, port, ip, zero)
        ai_ptr = UnsafePointer.address_of(ai).bitcast[sockaddr]()
        assert_true(libc.bind(fd, ai_ptr, sizeof[sockaddr_in]()) != -1)
        _ = ai
        assert_true(libc.listen(fd, C.int(0)) != -1)
        assert_true(libc.shutdown(fd, SHUT_RDWR) != -1)


def test_dynamic_bind_listen():
    _test_bind_listen(Libc[static=False]())


def test_static_bind_listen():
    _test_bind_listen(Libc[static=True]())


# TODO: needs counterpart/async test
# def _test_accept(libc: Libc):
#     ...


# def test_dynamic_accept():
#     ...


# def test_static_accept():
#     ...


# TODO: needs counterpart/async test
# def _test_connect(libc: Libc):
#     ...


# def test_dynamic_connect():
#     ...


# def test_static_connect():
#     ...


# TODO: needs counterpart/async test
# def _test_recv(libc: Libc):
#     ...


# def test_dynamic_recv():
#     ...


# def test_static_recv():
#     ...


# TODO: needs counterpart/async test
# def _test_recvfrom(libc: Libc):
#     ...


# def test_dynamic_recvfrom():
#     ...


# def test_static_recvfrom():
#     ...


# TODO: needs counterpart/async test
# def _test_send(libc: Libc):
#     ...


# def test_dynamic_send():
#     ...


# def test_static_send():
#     ...


# TODO: needs counterpart/async test
# def _test_sendto(libc: Libc):
#     ...


# def test_dynamic_sendto():
#     ...


# def test_static_sendto():
#     ...


def _test_getaddrinfo(libc: Libc):
    hints = addrinfo()
    hints.ai_family = AF_INET6
    hints.ai_socktype = SOCK_STREAM
    hints.ai_flags = 0
    hints.ai_protocol = IPPROTO_IP
    hints_p = UnsafePointer[addrinfo].address_of(hints)

    result = addrinfo()
    res_p = C.ptr_addr(int(UnsafePointer[addrinfo].address_of(result)))
    res_p_p = UnsafePointer[C.ptr_addr].address_of(res_p)

    err = libc.getaddrinfo(
        char_ptr("google.com"), char_ptr("443"), hints_p, res_p_p
    )
    assert_equal(err, 0)


def test_dynamic_getaddrinfo():
    _test_getaddrinfo(Libc[static=False]())


def test_static_getaddrinfo():
    _test_getaddrinfo(Libc[static=True]())


alias error_message = (
    (EAI_BADFLAGS, "Bad value for ai_flags"),
    (EAI_NONAME, "Name or service not known"),
    (EAI_AGAIN, "Temporary failure in name resolution"),
    (EAI_FAIL, "Non-recoverable failure in name resolution"),
    (EAI_NODATA, "No address associated with hostname"),
    (EAI_FAMILY, "ai_family not supported"),
    (EAI_SOCKTYPE, "ai_socktype not supported"),
    (EAI_SERVICE, "Servname not supported for ai_socktype"),
    (EAI_ADDRFAMILY, "Address family for hostname not supported"),
    (EAI_MEMORY, "Memory allocation failure"),
    (EAI_SYSTEM, "System error"),
    # (EAI_BADHINTS, "Bad value for hints"), # 'Unknown error' on Ubuntu 22.04
    # (EAI_PROTOCOL, "Resolved protocol is unknown"), # 'Unknown error' on Ubuntu 22.04
    # (EAI_OVERFLOW, "Argument buffer overflow"), # 'Unknown error' on Ubuntu 22.04
)


def _test_gai_strerror(libc: Libc):
    @parameter
    for i in range(len(error_message)):
        errno_msg = error_message.get[i, Tuple[Int, StringLiteral]]()
        errno = errno_msg.get[0, Int]()
        msg = errno_msg.get[1, StringLiteral]()
        res = char_ptr_to_string(libc.gai_strerror(errno))
        assert_equal(res, msg)


def test_dynamic_gai_strerror():
    _test_gai_strerror(Libc[static=False]())


def test_static_gai_strerror():
    _test_gai_strerror(Libc[static=True]())


def main():
    test_dynamic_htonl()
    test_static_htonl()
    test_dynamic_htons()
    test_static_htons()
    test_dynamic_ntohl()
    test_static_ntohl()
    test_dynamic_ntohs()
    test_static_ntohs()
    test_dynamic_inet_ntop()
    test_static_inet_ntop()
    test_dynamic_inet_pton()
    test_static_inet_pton()
    test_dynamic_inet_addr()
    test_static_inet_addr()
    test_dynamic_inet_aton()
    test_static_inet_aton()
    test_dynamic_inet_ntoa()
    test_static_inet_ntoa()
    test_dynamic_socket_create()
    test_static_socket_create()
    test_dynamic_socketpair()
    test_static_socketpair()
    test_dynamic_setsockopt()
    test_static_setsockopt()
    test_dynamic_bind_listen()
    test_static_bind_listen()
    # test_dynamic_accept()
    # test_static_accept()
    # test_dynamic_connect()
    # test_static_connect()
    # test_dynamic_recv()
    # test_static_recv()
    # test_dynamic_recvfrom()
    # test_static_recvfrom()
    # test_dynamic_send()
    # test_static_send()
    # test_dynamic_sendto()
    # test_static_sendto()
    # test_dynamic_shutdown()
    # test_static_shutdown()
    test_dynamic_getaddrinfo()
    test_static_getaddrinfo()
    test_dynamic_gai_strerror()
    test_static_gai_strerror()
