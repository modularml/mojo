"""Libc POSIX constants."""


# ===----------------------------------------------------------------------=== #
# Error constants (errno.h)
# ===----------------------------------------------------------------------=== #

alias SUCCESS = 0
"""Success."""
alias EPERM = 1
"""Operation not permitted."""
alias ENOENT = 2
"""No such file or directory."""
alias ESRCH = 3
"""No such process."""
alias EINTR = 4
"""Interrupted system call."""
alias EIO = 5
"""Input/output error."""
alias ENXIO = 6
"""No such device or address."""
alias E2BIG = 7
"""Argument list too long."""
alias ENOEXEC = 8
"""Exec format error."""
alias EBADF = 9
"""Bad file descriptor."""
alias ECHILD = 10
"""No child processes."""
alias EAGAIN = 11
"""Resource temporarily unavailable."""
alias EWOULDBLOCK = EAGAIN
"""Constant: EWOULDBLOCK."""
alias ENOMEM = 12
"""Cannot allocate memory."""
alias EACCES = 13
"""Permission denied."""
alias EFAULT = 14
"""Bad address."""
alias ENOTBLK = 15
"""Block device required."""
alias EBUSY = 16
"""Device or resource busy."""
alias EEXIST = 17
"""File exists."""
alias EXDEV = 18
"""Invalid cross-device link."""
alias ENODEV = 19
"""No such device."""
alias ENOTDIR = 20
"""Not a directory."""
alias EISDIR = 21
"""Is a directory."""
alias EINVAL = 22
"""Invalid argument."""
alias ENFILE = 23
"""Too many open files in system."""
alias EMFILE = 24
"""Too many open files."""
alias ENOTTY = 25
"""Inappropriate ioctl for device."""
alias ETXTBSY = 26
"""Text file busy."""
alias EFBIG = 27
"""File too large."""
alias ENOSPC = 28
"""No space left on device."""
alias ESPIPE = 29
"""Illegal seek."""
alias EROFS = 30
"""Read-only file system."""
alias EMLINK = 31
"""Too many links."""
alias EPIPE = 32
"""Broken pipe."""
alias EDOM = 33
"""Numerical argument out of domain."""
alias ERANGE = 34
"""Numerical result out of range."""
alias EDEADLK = 35
"""Resource deadlock avoided."""
alias ENAMETOOLONG = 36
"""File name too long."""
alias ENOLCK = 37
"""No locks available."""
alias ENOSYS = 38
"""Function not implemented."""
alias ENOTEMPTY = 39
"""Directory not empty."""
alias ELOOP = 40
"""Too many levels of symbolic links."""
alias ENOMSG = 42
"""No message of desired type."""
alias EIDRM = 43
"""Identifier removed."""
alias ECHRNG = 44
"""Channel number out of range."""
alias EL2NSYNC = 45
"""Level 2 not synchronized."""
alias EL3HLT = 46
"""Level 3 halted."""
alias EL3RST = 47
"""Level 3 reset."""
alias ELNRNG = 48
"""Link number out of range."""
alias EUNATCH = 49
"""Protocol driver not attached."""
alias ENOCSI = 50
"""No CSI structure available."""
alias EL2HLT = 51
"""Level 2 halted."""
alias EBADE = 52
"""Invalid exchange."""
alias EBADR = 53
"""Invalid request descriptor."""
alias EXFULL = 54
"""Exchange full."""
alias ENOANO = 55
"""No anode."""
alias EBADRQC = 56
"""Invalid request code."""
alias EBADSLT = 57
"""Invalid slot."""
alias EBFONT = 59
"""Bad font file format."""
alias ENOSTR = 60
"""Device not a stream."""
alias ENODATA = 61
"""No data available."""
alias ETIME = 62
"""Timer expired."""
alias ENOSR = 63
"""Out of streams resources."""
alias ENONET = 64
"""Machine is not on the network."""
alias ENOPKG = 65
"""Package not installed."""
alias EREMOTE = 66
"""Object is remote."""
alias ENOLINK = 67
"""Link has been severed."""
alias EADV = 68
"""Advertise error."""
alias ESRMNT = 69
"""Srmount error."""
alias ECOMM = 70
"""Communication error on send."""
alias EPROTO = 71
"""Protocol error."""
alias EMULTIHOP = 72
"""Multihop attempted."""
alias EDOTDOT = 73
"""RFS specific error."""
alias EBADMSG = 74
"""Bad message."""
alias EOVERFLOW = 75
"""Value too large for defined data type."""
alias ENOTUNIQ = 76
"""Name not unique on network."""
alias EBADFD = 77
"""File descriptor in bad state."""
alias EREMCHG = 78
"""Remote address changed."""
alias ELIBACC = 79
"""Can not access a needed shared library."""
alias ELIBBAD = 80
"""Accessing a corrupted shared library."""
alias ELIBSCN = 81
""".lib section in a.out corrupted."""
alias ELIBMAX = 82
"""Attempting to link in too many shared libraries."""
alias ELIBEXEC = 83
"""Cannot exec a shared library directly."""
alias EILSEQ = 84
"""Invalid or incomplete multibyte or wide character."""
alias ERESTART = 85
"""Interrupted system call should be restarted."""
alias ESTRPIPE = 86
"""Streams pipe error."""
alias EUSERS = 87
"""Too many users."""
alias ENOTSOCK = 88
"""Socket operation on non-socket."""
alias EDESTADDRREQ = 89
"""Destination address required."""
alias EMSGSIZE = 90
"""Message too long."""
alias EPROTOTYPE = 91
"""Protocol wrong type for socket."""
alias ENOPROTOOPT = 92
"""Protocol not available."""
alias EPROTONOSUPPORT = 93
"""Protocol not supported."""
alias ESOCKTNOSUPPORT = 94
"""Socket type not supported."""
alias EOPNOTSUPP = 95
"""Operation not supported."""
alias EPFNOSUPPORT = 96
"""Protocol family not supported."""
alias EAFNOSUPPORT = 97
"""Address family not supported by protocol."""
alias EADDRINUSE = 98
"""Address already in use."""
alias EADDRNOTAVAIL = 99
"""Cannot assign requested address."""
alias ENETDOWN = 100
"""Network is down."""
alias ENETUNREACH = 101
"""Network is unreachable."""
alias ENETRESET = 102
"""Network dropped connection on reset."""
alias ECONNABORTED = 103
"""Software caused connection abort."""
alias ECONNRESET = 104
"""Connection reset by peer."""
alias ENOBUFS = 105
"""No buffer space available."""
alias EISCONN = 106
"""Transport endpoint is already connected."""
alias ENOTCONN = 107
"""Transport endpoint is not connected."""
alias ESHUTDOWN = 108
"""Cannot send after transport endpoint shutdown."""
alias ETOOMANYREFS = 109
"""Too many references: cannot splice."""
alias ETIMEDOUT = 110
"""Connection timed out."""
alias ECONNREFUSED = 111
"""Connection refused."""
alias EHOSTDOWN = 112
"""Host is down."""
alias EHOSTUNREACH = 113
"""No route to host."""
alias EALREADY = 114
"""Operation already in progress."""
alias EINPROGRESS = 115
"""Operation now in progress."""
alias ESTALE = 116
"""Stale file handle."""
alias EUCLEAN = 117
"""Structure needs cleaning."""
alias ENOTNAM = 118
"""Not a XENIX named type file."""
alias ENAVAIL = 119
"""No XENIX semaphores available."""
alias EISNAM = 120
"""Is a named type file."""
alias EREMOTEIO = 121
"""Remote I/O error."""
alias EDQUOT = 122
"""Disk quota exceeded."""
alias ENOMEDIUM = 123
"""No medium found."""
alias EMEDIUMTYPE = 124
"""Wrong medium type."""
alias ECANCELED = 125
"""Operation canceled."""
alias ENOKEY = 126
"""Required key not available."""
alias EKEYEXPIRED = 127
"""Key has expired."""
alias EKEYREVOKED = 128
"""Key has been revoked."""
alias EKEYREJECTED = 129
"""Key was rejected by service."""
alias EOWNERDEAD = 130
"""Owner died."""
alias ENOTRECOVERABLE = 131
"""State not recoverable."""
alias ERFKILL = 132
"""Operation not possible due to RF-kill."""
alias EHWPOISON = 133
"""Memory page has hardware error."""


# ===----------------------------------------------------------------------=== #
# Networking constants
# ===----------------------------------------------------------------------=== #

# Address Family Constants
alias AF_UNSPEC = 0
"""Constant: AF_UNSPEC."""
alias AF_UNIX = 1
"""Constant: AF_UNIX."""
alias AF_LOCAL = AF_UNIX
"""Constant: AF_LOCAL."""
alias AF_INET = 2
"""Constant: AF_INET."""
alias AF_AX25 = 3
"""Constant: AF_AX25."""
alias AF_IPX = 4
"""Constant: AF_IPX."""
alias AF_APPLETALK = 5
"""Constant: AF_APPLETALK."""
alias AF_NETROM = 6
"""Constant: AF_NETROM."""
alias AF_BRIDGE = 7
"""Constant: AF_BRIDGE."""
alias AF_ATMPVC = 8
"""Constant: AF_ATMPVC."""
alias AF_X25 = 9
"""Constant: AF_X25."""
alias AF_INET6 = 10
"""Constant: AF_INET6."""
alias AF_ROSE = 11
"""Constant: AF_ROSE."""
alias AF_DECnet = 12
"""Constant: AF_DECnet."""
alias AF_NETBEUI = 13
"""Constant: AF_NETBEUI."""
alias AF_SECURITY = 14
"""Constant: AF_SECURITY."""
alias AF_KEY = 15
"""Constant: AF_KEY."""
alias AF_NETLINK = 16
"""Constant: AF_NETLINK."""
alias AF_ROUTE = AF_NETLINK
"""Constant: AF_ROUTE."""
alias AF_PACKET = 17
"""Constant: AF_PACKET."""
alias AF_ASH = 18
"""Constant: AF_ASH."""
alias AF_ECONET = 19
"""Constant: AF_ECONET."""
alias AF_ATMSVC = 20
"""Constant: AF_ATMSVC."""
alias AF_RDS = 21
"""Constant: AF_RDS."""
alias AF_SNA = 22
"""Constant: AF_SNA."""
alias AF_IRDA = 23
"""Constant: AF_IRDA."""
alias AF_PPPOX = 24
"""Constant: AF_PPPOX."""
alias AF_WANPIPE = 25
"""Constant: AF_WANPIPE."""
alias AF_LLC = 26
"""Constant: AF_LLC."""
alias AF_CAN = 29
"""Constant: AF_CAN."""
alias AF_TIPC = 30
"""Constant: AF_TIPC."""
alias AF_BLUETOOTH = 31
"""Constant: AF_BLUETOOTH."""
alias AF_IUCV = 32
"""Constant: AF_IUCV."""
alias AF_RXRPC = 33
"""Constant: AF_RXRPC."""
alias AF_ISDN = 34
"""Constant: AF_ISDN."""
alias AF_PHONET = 35
"""Constant: AF_PHONET."""
alias AF_IEEE802154 = 36
"""Constant: AF_IEEE802154."""
alias AF_CAIF = 37
"""Constant: AF_CAIF."""
alias AF_ALG = 38
"""Constant: AF_ALG."""
alias AF_NFC = 39
"""Constant: AF_NFC."""
alias AF_VSOCK = 40
"""Constant: AF_VSOCK."""
alias AF_KCM = 41
"""Constant: AF_KCM."""
alias AF_QIPCRTR = 42
"""Constant: AF_QIPCRTR."""
alias AF_MAX = 43
"""Constant: AF_MAX."""

alias PF_UNSPEC = AF_UNSPEC
"""Constant: PF_UNSPEC."""
alias PF_UNIX = AF_UNIX
"""Constant: PF_UNIX."""
alias PF_LOCAL = AF_LOCAL
"""Constant: PF_LOCAL."""
alias PF_INET = AF_INET
"""Constant: PF_INET."""
alias PF_AX25 = AF_AX25
"""Constant: PF_AX25."""
alias PF_IPX = AF_IPX
"""Constant: PF_IPX."""
alias PF_APPLETALK = AF_APPLETALK
"""Constant: PF_APPLETALK."""
alias PF_NETROM = AF_NETROM
"""Constant: PF_NETROM."""
alias PF_BRIDGE = AF_BRIDGE
"""Constant: PF_BRIDGE."""
alias PF_ATMPVC = AF_ATMPVC
"""Constant: PF_ATMPVC."""
alias PF_X25 = AF_X25
"""Constant: PF_X25."""
alias PF_INET6 = AF_INET6
"""Constant: PF_INET6."""
alias PF_ROSE = AF_ROSE
"""Constant: PF_ROSE."""
alias PF_DECnet = AF_DECnet
"""Constant: PF_DECnet."""
alias PF_NETBEUI = AF_NETBEUI
"""Constant: PF_NETBEUI."""
alias PF_SECURITY = AF_SECURITY
"""Constant: PF_SECURITY."""
alias PF_KEY = AF_KEY
"""Constant: PF_KEY."""
alias PF_NETLINK = AF_NETLINK
"""Constant: PF_NETLINK."""
alias PF_ROUTE = AF_ROUTE
"""Constant: PF_ROUTE."""
alias PF_PACKET = AF_PACKET
"""Constant: PF_PACKET."""
alias PF_ASH = AF_ASH
"""Constant: PF_ASH."""
alias PF_ECONET = AF_ECONET
"""Constant: PF_ECONET."""
alias PF_ATMSVC = AF_ATMSVC
"""Constant: PF_ATMSVC."""
alias PF_RDS = AF_RDS
"""Constant: PF_RDS."""
alias PF_SNA = AF_SNA
"""Constant: PF_SNA."""
alias PF_IRDA = AF_IRDA
"""Constant: PF_IRDA."""
alias PF_PPPOX = AF_PPPOX
"""Constant: PF_PPPOX."""
alias PF_WANPIPE = AF_WANPIPE
"""Constant: PF_WANPIPE."""
alias PF_LLC = AF_LLC
"""Constant: PF_LLC."""
alias PF_CAN = AF_CAN
"""Constant: PF_CAN."""
alias PF_TIPC = AF_TIPC
"""Constant: PF_TIPC."""
alias PF_BLUETOOTH = AF_BLUETOOTH
"""Constant: PF_BLUETOOTH."""
alias PF_IUCV = AF_IUCV
"""Constant: PF_IUCV."""
alias PF_RXRPC = AF_RXRPC
"""Constant: PF_RXRPC."""
alias PF_ISDN = AF_ISDN
"""Constant: PF_ISDN."""
alias PF_PHONET = AF_PHONET
"""Constant: PF_PHONET."""
alias PF_IEEE802154 = AF_IEEE802154
"""Constant: PF_IEEE802154."""
alias PF_CAIF = AF_CAIF
"""Constant: PF_CAIF."""
alias PF_ALG = AF_ALG
"""Constant: PF_ALG."""
alias PF_NFC = AF_NFC
"""Constant: PF_NFC."""
alias PF_VSOCK = AF_VSOCK
"""Constant: PF_VSOCK."""
alias PF_KCM = AF_KCM
"""Constant: PF_KCM."""
alias PF_QIPCRTR = AF_QIPCRTR
"""Constant: PF_QIPCRTR."""
alias PF_MAX = AF_MAX
"""Constant: PF_MAX."""

# Socket Type constants
alias SOCK_STREAM = 1
"""Constant: SOCK_STREAM."""
alias SOCK_DGRAM = 2
"""Constant: SOCK_DGRAM."""
alias SOCK_RAW = 3
"""Constant: SOCK_RAW."""
alias SOCK_RDM = 4
"""Constant: SOCK_RDM."""
alias SOCK_SEQPACKET = 5
"""Constant: SOCK_SEQPACKET."""
alias SOCK_DCCP = 6
"""Constant: SOCK_DCCP."""
alias SOCK_PACKET = 10
"""Constant: SOCK_PACKET."""
# alias SOCK_CLOEXEC = O_CLOEXEC
# alias SOCK_NONBLOCK = O_NONBLOCK

# Internet (IP) protocols
# Updated from http://www.iana.org/assignments/protocol-numbers and other
# sources.
alias IPPROTO_IP = 0
"""internet protocol, pseudo protocol number."""
alias IPPROTO_HOPOPT = 0
"""IPv6 Hop-by-Hop Option [RFC1883]."""
alias IPPROTO_ICMP = 1
"""internet control message protocol."""
alias IPPROTO_IGMP = 2
"""Internet Group Management."""
alias IPPROTO_GGP = 3
"""gateway-gateway protocol."""
alias IPPROTO_IP_ENCAP = 4
"""IP encapsulated in IP (officially ``IP'')."""
alias IPPROTO_ST = 5
"""ST datagram mode."""
alias IPPROTO_TCP = 6
"""transmission control protocol."""
alias IPPROTO_EGP = 8
"""exterior gateway protocol."""
alias IPPROTO_IGP = 9
"""any private interior gateway (Cisco)."""
alias IPPROTO_PUP = 12
"""PARC universal packet protocol."""
alias IPPROTO_UDP = 17
"""user datagram protocol."""
alias IPPROTO_HMP = 20
"""host monitoring protocol."""
alias IPPROTO_XNS_IDP = 22
"""Xerox NS IDP."""
alias IPPROTO_RDP = 27
""""reliable datagram" protocol."""
alias IPPROTO_ISO_TP4 = 29
"""ISO Transport Protocol class 4 [RFC905]."""
alias IPPROTO_DCCP = 33
"""Datagram Congestion Control Prot. [RFC4340]."""
alias IPPROTO_XTP = 36
"""Xpress Transfer Protocol."""
alias IPPROTO_DDP = 37
"""Datagram Delivery Protocol."""
alias IPPROTO_IDPR_CMTP = 38
"""IDPR Control Message Transport."""
alias IPPROTO_IPV6 = 41
"""Internet Protocol, version 6."""
alias IPPROTO_IPV6_ROUTE = 43
"""Routing Header for IPv6."""
alias IPPROTO_IPV6_FRAG = 44
"""Fragment Header for IPv6."""
alias IPPROTO_IDRP = 45
"""Inter_Domain Routing Protocol."""
alias IPPROTO_RSVP = 46
"""Reservation Protocol."""
alias IPPROTO_GRE = 47
"""General Routing Encapsulation."""
alias IPPROTO_IPSEC_ESP = 50
"""Encap Security Payload [RFC2406]."""
alias IPPROTO_IPSEC_AH = 51
"""Authentication Header [RFC2402]."""
alias IPPROTO_SKIP = 57
"""SKIP."""
alias IPPROTO_IPV6_ICMP = 58
"""ICMP for IPv6."""
alias IPPROTO_IPV6_NONXT = 59
"""No Next Header for IPv6."""
alias IPPROTO_IPV6_OPTS = 60
"""Destination Options for IPv6."""
alias IPPROTO_RSPF_CPHB = 73
"""Radio Shortest Path First (officially CPHB)."""
alias IPPROTO_VMTP = 81
"""Versatile Message Transport."""
alias IPPROTO_EIGRP = 88
"""Enhanced Interior Routing Protocol (Cisco)."""
alias IPPROTO_OSPFIGP = 89
"""Open Shortest Path First IGP."""
alias IPPROTO_AX_25 = 93
"""AX.25 frames."""
alias IPPROTO_IPIP = 94
"""IP_within_IP Encapsulation Protocol."""
alias IPPROTO_ETHERIP = 97
"""Ethernet_within_IP Encapsulation [RFC3378]."""
alias IPPROTO_ENCAP = 98
"""Yet Another IP encapsulation [RFC1241]."""
alias IPPROTO_PIM = 103
"""Protocol Independent Multicast."""
alias IPPROTO_IPCOMP = 108
"""IP Payload Compression Protocol."""
alias IPPROTO_VRRP = 112
"""Virtual Router Redundancy Protocol [RFC5798]."""
alias IPPROTO_L2TP = 115
"""Layer Two Tunneling Protocol [RFC2661]."""
alias IPPROTO_ISIS = 124
"""IS_IS over IPv4."""
alias IPPROTO_SCTP = 132
"""Stream Control Transmission Protocol."""
alias IPPROTO_FC = 133
"""Fibre Channel."""
alias IPPROTO_MOBILITY_HEADER = 135
"""Mobility Support for IPv6 [RFC3775]."""
alias IPPROTO_UDPLITE = 136
"""UDP_Lite [RFC3828]."""
alias IPPROTO_MPLS_IN_IP = 137
"""MPLS_in_IP [RFC4023]."""
alias IPPROTO_HIP = 139
"""Host Identity Protocol."""
alias IPPROTO_SHIM6 = 140
"""Shim6 Protocol [RFC5533]."""
alias IPPROTO_WESP = 141
"""Wrapped Encapsulating Security Payload."""
alias IPPROTO_ROHC = 142
"""Robust Header Compression."""
alias IPPROTO_RAW = 255
"""Raw IP packets."""

# Address Information
alias AI_PASSIVE = 1
"""Constant: AI_PASSIVE."""
alias AI_CANONNAME = 2
"""Constant: AI_CANONNAME."""
alias AI_NUMERICHOST = 4
"""Constant: AI_NUMERICHOST."""
alias AI_V4MAPPED = 8
"""Constant: AI_V4MAPPED."""
alias AI_ALL = 16
"""Constant: AI_ALL."""
alias AI_ADDRCONFIG = 32
"""Constant: AI_ADDRCONFIG."""
alias AI_IDN = 64
"""Constant: AI_IDN."""

alias INET_ADDRSTRLEN = 16
"""Constant: INET_ADDRSTRLEN."""
alias INET6_ADDRSTRLEN = 46
"""Constant: INET6_ADDRSTRLEN."""

alias SHUT_RD = 0
"""Constant: SHUT_RD."""
alias SHUT_WR = 1
"""Constant: SHUT_WR."""
alias SHUT_RDWR = 2
"""Constant: SHUT_RDWR."""

# Socket level options (SOL_SOCKET)
alias SOL_SOCKET = 1
"""Constant: SOL_SOCKET."""

alias SO_DEBUG = 1
"""Constant: SO_DEBUG."""
alias SO_REUSEADDR = 2
"""Constant: SO_REUSEADDR."""
alias SO_TYPE = 3
"""Constant: SO_TYPE."""
alias SO_ERROR = 4
"""Constant: SO_ERROR."""
alias SO_DONTROUTE = 5
"""Constant: SO_DONTROUTE."""
alias SO_BROADCAST = 6
"""Constant: SO_BROADCAST."""
alias SO_SNDBUF = 7
"""Constant: SO_SNDBUF."""
alias SO_RCVBUF = 8
"""Constant: SO_RCVBUF."""
alias SO_KEEPALIVE = 9
"""Constant: SO_KEEPALIVE."""
alias SO_OOBINLINE = 10
"""Constant: SO_OOBINLINE."""
alias SO_NO_CHECK = 11
"""Constant: SO_NO_CHECK."""
alias SO_PRIORITY = 12
"""Constant: SO_PRIORITY."""
alias SO_LINGER = 13
"""Constant: SO_LINGER."""
alias SO_BSDCOMPAT = 14
"""Constant: SO_BSDCOMPAT."""
alias SO_REUSEPORT = 15
"""Constant: SO_REUSEPORT."""
alias SO_PASSCRED = 16
"""Constant: SO_PASSCRED."""
alias SO_PEERCRED = 17
"""Constant: SO_PEERCRED."""
alias SO_RCVLOWAT = 18
"""Constant: SO_RCVLOWAT."""
alias SO_SNDLOWAT = 19
"""Constant: SO_SNDLOWAT."""
alias SO_RCVTIMEO = 20
"""Constant: SO_RCVTIMEO."""
alias SO_SNDTIMEO = 21
"""Constant: SO_SNDTIMEO."""
alias SO_SECURITY_AUTHENTICATION = 22
"""Constant: SO_SECURITY_AUTHENTICATION."""
alias SO_SECURITY_ENCRYPTION_TRANSPORT = 23
"""Constant: SO_SECURITY_ENCRYPTION_TRANSPORT."""
alias SO_SECURITY_ENCRYPTION_NETWORK = 24
"""Constant: SO_SECURITY_ENCRYPTION_NETWORK."""
alias SO_BINDTODEVICE = 25
"""Constant: SO_BINDTODEVICE."""
alias SO_ATTACH_FILTER = 26
"""Constant: SO_ATTACH_FILTER."""
alias SO_DETACH_FILTER = 27
"""Constant: SO_DETACH_FILTER."""
alias SO_GET_FILTER = SO_ATTACH_FILTER
"""Constant: SO_GET_FILTER."""
alias SO_PEERNAME = 28
"""Constant: SO_PEERNAME."""
alias SO_TIMESTAMP = 29
"""Constant: SO_TIMESTAMP."""
alias SO_ACCEPTCONN = 30
"""Constant: SO_ACCEPTCONN."""
alias SO_PEERSEC = 31
"""Constant: SO_PEERSEC."""
alias SO_SNDBUFFORCE = 32
"""Constant: SO_SNDBUFFORCE."""
alias SO_RCVBUFFORCE = 33
"""Constant: SO_RCVBUFFORCE."""
alias SO_PASSSEC = 34
"""Constant: SO_PASSSEC."""
alias SO_TIMESTAMPNS = 35
"""Constant: SO_TIMESTAMPNS."""
alias SO_MARK = 36
"""Constant: SO_MARK."""
alias SO_TIMESTAMPING = 37
"""Constant: SO_TIMESTAMPING."""
alias SO_PROTOCOL = 38
"""Constant: SO_PROTOCOL."""
alias SO_DOMAIN = 39
"""Constant: SO_DOMAIN."""
alias SO_RXQ_OVFL = 40
"""Constant: SO_RXQ_OVFL."""
alias SO_WIFI_STATUS = 41
"""Constant: SO_WIFI_STATUS."""
alias SCM_WIFI_STATUS = SO_WIFI_STATUS
"""Constant: SCM_WIFI_STATUS."""
alias SO_PEEK_OFF = 42
"""Constant: SO_PEEK_OFF."""
alias SO_NOFCS = 43
"""Constant: SO_NOFCS."""
alias SO_LOCK_FILTER = 44
"""Constant: SO_LOCK_FILTER."""
alias SO_SELECT_ERR_QUEUE = 45
"""Constant: SO_SELECT_ERR_QUEUE."""
alias SO_BUSY_POLL = 46
"""Constant: SO_BUSY_POLL."""
alias SO_MAX_PACING_RATE = 47
"""Constant: SO_MAX_PACING_RATE."""
alias SO_BPF_EXTENSIONS = 48
"""Constant: SO_BPF_EXTENSIONS."""
alias SO_INCOMING_CPU = 49
"""Constant: SO_INCOMING_CPU."""
alias SO_ATTACH_BPF = 50
"""Constant: SO_ATTACH_BPF."""
alias SO_DETACH_BPF = SO_DETACH_FILTER
"""Constant: SO_DETACH_BPF."""
alias SO_ATTACH_REUSEPORT_CBPF = 51
"""Constant: SO_ATTACH_REUSEPORT_CBPF."""
alias SO_ATTACH_REUSEPORT_EBPF = 52
"""Constant: SO_ATTACH_REUSEPORT_EBPF."""
alias SO_CNX_ADVICE = 53
"""Constant: SO_CNX_ADVICE."""
alias SCM_TIMESTAMPING_OPT_STATS = 54
"""Constant: SCM_TIMESTAMPING_OPT_STATS."""
alias SO_MEMINFO = 55
"""Constant: SO_MEMINFO."""
alias SO_INCOMING_NAPI_ID = 56
"""Constant: SO_INCOMING_NAPI_ID."""
alias SO_COOKIE = 57
"""Constant: SO_COOKIE."""
alias SCM_TIMESTAMPING_PKTINFO = 58
"""Constant: SCM_TIMESTAMPING_PKTINFO."""
alias SO_PEERGROUPS = 59
"""Constant: SO_PEERGROUPS."""
alias SO_ZEROCOPY = 60
"""Constant: SO_ZEROCOPY."""
alias SO_TXTIME = 61
"""Constant: SO_TXTIME."""
alias SCM_TXTIME = SO_TXTIME
"""Constant: SCM_TXTIME."""
alias SO_BINDTOIFINDEX = 62
"""Constant: SO_BINDTOIFINDEX."""
alias SO_TIMESTAMP_NEW = 63
"""Constant: SO_TIMESTAMP_NEW."""
alias SO_TIMESTAMPNS_NEW = 64
"""Constant: SO_TIMESTAMPNS_NEW."""
alias SO_TIMESTAMPING_NEW = 65
"""Constant: SO_TIMESTAMPING_NEW."""
alias SO_RCVTIMEO_NEW = 66
"""Constant: SO_RCVTIMEO_NEW."""
alias SO_SNDTIMEO_NEW = 67
"""Constant: SO_SNDTIMEO_NEW."""
alias SO_DETACH_REUSEPORT_BPF = 68
"""Constant: SO_DETACH_REUSEPORT_BPF."""

# TCP level options (IPPROTO_TCP)
alias TCP_NODELAY = 1
"""Constant: TCP_NODELAY."""
alias TCP_KEEPIDLE = 2
"""Constant: TCP_KEEPIDLE."""
alias TCP_KEEPINTVL = 3
"""Constant: TCP_KEEPINTVL."""
alias TCP_KEEPCNT = 4
"""Constant: TCP_KEEPCNT."""

# IPv4 level options (IPPROTO_IP)
alias IP_TOS = 1
"""IP type of service and precedence."""
alias IP_TTL = 2
"""IP time to live."""
alias IP_HDRINCL = 3
"""Header is included with data."""
alias IP_OPTIONS = 4
"""IP per-packet options."""
alias IP_RECVOPTS = 6
"""Receive all IP options w/datagram."""
alias IP_RETOPTS = 7
"""Set/get IP per-packet options."""
alias IP_RECVRETOPTS = IP_RETOPTS
"""Receive IP options for response."""
alias IP_MULTICAST_IF = 32
"""Set/get IP multicast i/f."""
alias IP_MULTICAST_TTL = 33
"""Set/get IP multicast ttl."""
alias IP_MULTICAST_LOOP = 34
"""Set/get IP multicast loopback."""
alias IP_ADD_MEMBERSHIP = 35
"""Add an IP group membership."""
alias IP_DROP_MEMBERSHIP = 36
"""Drop an IP group membership."""
alias IP_UNBLOCK_SOURCE = 37
"""Unblock data from source."""
alias IP_BLOCK_SOURCE = 38
"""Block data from source."""
alias IP_ADD_SOURCE_MEMBERSHIP = 39
"""Join source group."""
alias IP_DROP_SOURCE_MEMBERSHIP = 40
"""Leave source group."""
alias IP_MSFILTER = 41

# IPv6 level options (IPPROTO_IPV6)
alias IPV6_ADDRFORM = 1
"""Constant: IPV6_ADDRFORM."""
alias IPV6_2292PKTINFO = 2
"""Constant: IPV6_2292PKTINFO."""
alias IPV6_2292HOPOPTS = 3
"""Constant: IPV6_2292HOPOPTS."""
alias IPV6_2292DSTOPTS = 4
"""Constant: IPV6_2292DSTOPTS."""
alias IPV6_2292RTHDR = 5
"""Constant: IPV6_2292RTHDR."""
alias IPV6_2292PKTOPTIONS = 6
"""Constant: IPV6_2292PKTOPTIONS."""
alias IPV6_CHECKSUM = 7
"""Constant: IPV6_CHECKSUM."""
alias IPV6_2292HOPLIMIT = 8
"""Constant: IPV6_2292HOPLIMIT."""
alias IPV6_NEXTHOP = 9
"""Constant: IPV6_NEXTHOP."""
alias IPV6_AUTHHDR = 10
"""Constant: IPV6_AUTHHDR."""
alias IPV6_UNICAST_HOPS = 16
"""Constant: Set the unicast hop limit for the socket."""
alias IPV6_MULTICAST_IF = 17
"""Constant: IPV6_MULTICAST_IF."""
alias IPV6_MULTICAST_HOPS = 18
"""Constant: Set the multicast hop limit for the socket."""
alias IPV6_MULTICAST_LOOP = 19
"""Constant: IPV6_MULTICAST_LOOP."""
alias IPV6_JOIN_GROUP = 20
"""Constant: Join IPv6 multicast group."""
alias IPV6_LEAVE_GROUP = 21
"""Constant: Leave IPv6 multicast group."""
alias IPV6_ROUTER_ALERT = 22
"""Constant: IPV6_ROUTER_ALERT."""
alias IPV6_MTU_DISCOVER = 23
"""Constant: IPV6_MTU_DISCOVER."""
alias IPV6_MTU = 24
"""Constant: IPV6_MTU."""
alias IPV6_RECVERR = 25
"""Constant: IPV6_RECVERR."""
alias IPV6_V6ONLY = 26
"""Constant: Don't support IPv4 access."""
alias IPV6_JOIN_ANYCAST = 27
"""Constant: IPV6_JOIN_ANYCAST."""
alias IPV6_LEAVE_ANYCAST = 28
"""Constant: IPV6_LEAVE_ANYCAST."""
alias IPV6_IPSEC_POLICY = 34
"""Constant: IPV6_IPSEC_POLICY."""
alias IPV6_XFRM_POLICY = 35
"""Constant: IPV6_XFRM_POLICY."""
alias IPV6_RECVPKTINFO = 49
"""Pass an IPV6_RECVPKTINFO ancillary message that contains a in6_pktinfo
structure that supplies some information about the incoming packet."""
alias IPV6_PKTINFO = 50
"""Constant: IPV6_PKTINFO."""
alias IPV6_RECVHOPLIMIT = 51
"""Constant: IPV6_RECVHOPLIMIT."""
alias IPV6_HOPLIMIT = 52
"""Constant: IPV6_HOPLIMIT."""
alias IPV6_RECVHOPOPTS = 53
"""Constant: IPV6_RECVHOPOPTS."""
alias IPV6_HOPOPTS = 54
"""Constant: IPV6_HOPOPTS."""
alias IPV6_RTHDRDSTOPTS = 55
"""Constant: IPV6_RTHDRDSTOPTS."""
alias IPV6_RECVRTHDR = 56
"""Constant: IPV6_RECVRTHDR."""
alias IPV6_RTHDR = 57
"""Constant: IPV6_RTHDR."""
alias IPV6_RECVDSTOPTS = 58
"""Constant: IPV6_RECVDSTOPTS."""
alias IPV6_DSTOPTS = 59
"""Constant: IPV6_DSTOPTS."""
alias IPV6_RECVTCLASS = 66
"""Constant: IPV6_RECVTCLASS."""
alias IPV6_TCLASS = 67
"""Constant: IPV6_TCLASS."""
alias IPV6_ADDR_PREFERENCES = 72
"""RFC5014: Source address selection."""
alias IPV6_PREFER_SRC_TMP = 0x0001
"""Prefer temporary address as source."""
alias IPV6_PREFER_SRC_PUBLIC = 0x0002
"""Prefer public address as source."""
alias IPV6_PREFER_SRC_PUBTMP_DEFAULT = 0x0100
"""Either public or temporary address is selected as a default source depending
on the output interface configuration (this is the default value)."""
alias IPV6_PREFER_SRC_COA = 0x0004
"""Prefer Care-of address as source."""
alias IPV6_PREFER_SRC_HOME = 0x0400
"""Prefer Home address as source."""
alias IPV6_PREFER_SRC_CGA = 0x0008
"""Prefer CGA (Cryptographically Generated Address) address as source."""
alias IPV6_PREFER_SRC_NONCGA = 0x0800
"""Prefer non-CGA address as source."""

# Obsolete synonyms for the above.
alias IPV6_ADD_MEMBERSHIP = IPV6_JOIN_GROUP
"""Constant: IPV6_ADD_MEMBERSHIP."""
alias IPV6_DROP_MEMBERSHIP = IPV6_LEAVE_GROUP
"""Constant: IPV6_DROP_MEMBERSHIP."""
alias IPV6_RXHOPOPTS = IPV6_HOPOPTS
"""Constant: IPV6_RXHOPOPTS."""
alias IPV6_RXDSTOPTS = IPV6_DSTOPTS
"""Constant: IPV6_RXDSTOPTS."""

# IPV6_MTU_DISCOVER values.
alias IPV6_PMTUDISC_DONT = 0
"""Never send DF frames."""
alias IPV6_PMTUDISC_WANT = 1
"""Use per route hints."""
alias IPV6_PMTUDISC_DO = 2
"""Always DF."""
alias IPV6_PMTUDISC_PROBE = 3
"""Ignore dst pmtu."""

# netdb.h
alias EAI_BADFLAGS = -1
"""Bad value for ai_flags."""
alias EAI_NONAME = -2
"""Name or service not known."""
alias EAI_AGAIN = -3
"""Temporary failure in name resolution."""
alias EAI_FAIL = -4
"""Non-recoverable failure in name resolution."""
alias EAI_NODATA = -5
"""No address associated with hostname."""
alias EAI_FAMILY = -6
"""Error: ai_family not supported."""
alias EAI_SOCKTYPE = -7
"""Error: ai_socktype not supported."""
alias EAI_SERVICE = -8
"""Servname not supported for ai_socktype."""
alias EAI_ADDRFAMILY = -9
"""Address family for hostname not supported."""
alias EAI_MEMORY = -10
"""Memory allocation failure."""
alias EAI_SYSTEM = -11
"""System error."""
alias EAI_BADHINTS = -12
"""Bad value for hints."""
alias EAI_PROTOCOL = -13
"""Resolved protocol is unknown."""
alias EAI_OVERFLOW = -14
"""Argument buffer overflow."""

# ===----------------------------------------------------------------------=== #
# File constants (stdio.h, fcntl.h, etc.)
# ===----------------------------------------------------------------------=== #

alias EOF = -1
"""Constant: EOF."""

alias STDIN_FILENO = 0
"""Constant: STDIN_FILENO."""
alias STDOUT_FILENO = 1
"""Constant: STDOUT_FILENO."""
alias STDERR_FILENO = 2
"""Constant: STDERR_FILENO."""

alias FM_READ = "r"
"""Open text file for reading. The stream is positioned at the beginning of the
file."""
alias FM_READ_WRITE = "r+"
"""Open for reading and writing. The stream is positioned at the beginning of
the file.
"""
alias FM_WRITE = "w"
"""Truncate file to zero length or create text file for writing. The stream is
positioned at the beginning of the file.
"""
alias FM_WRITE_READ_CREATE = "w+"
"""Open for reading and writing. The file is created if it does not exist,
otherwise it is truncated. The stream is positioned at the beginning of the
file.
"""
alias FM_APPEND = "a"
"""Open for appending (writing at end of file). The file is created if it does
not exist. The stream is positioned at the end of the file.
"""
alias FM_APPEND_READ = "a+"
"""Open for reading and appending (writing at end of file). The file is created
if it does not exist. The initial file position for reading is at the beginning
of the file, but output is always appended to the end of the file.
"""
# NOTE: The mode string can also include the letter 'b' either as a last
# character or as a character between the characters in any of the two-character
# strings described above. This is strictly for compatibility with C89 and has
# no effect; the 'b' is ignored on all POSIX conforming systems, including
# Linux. (Other systems may treat text files and binary files differently, and
# adding the 'b' may be a good idea if you do I/O to a binary file and expect
# that your program may be ported to non-UNIX environments).


alias SEEK_SET = 0
"""Constant: SEEK_SET."""
alias SEEK_CUR = 1
"""Constant: SEEK_CUR."""
alias SEEK_END = 2
"""Constant: SEEK_END."""

alias O_RDONLY = 0o0
"""Open for reading only."""
alias O_WRONLY = 0o1
"""Open for writing only."""
alias O_RDWR = 0o2
"""Open for reading and writing."""
alias O_ACCMODE = 0o3
"""Constant: O_ACCMODE."""
alias O_APPEND = 0o2000
"""Set append mode."""
alias O_CREAT = 0o100
"""Create file if it does not exist."""
alias O_TRUNC = 0o1000
"""If the file exists and is a regular file, and the file is successfully opened
O_RDWR or O_WRONLY, its length shall be truncated to 0, and the mode and owner
shall be unchanged. It shall have no effect on FIFO special files or terminal
device files. Its effect on other file types is implementation-defined. The
result of using O_TRUNC without either O_RDWR or O_WRONLY is undefined."""
alias O_EXCL = 0o200
"""If O_CREAT and O_EXCL are set, open() shall fail if the file exists."""
alias O_SYNC = 0o10000
"""Write I/O operations on the file descriptor shall complete as defined by
synchronized I/O file integrity completion."""
alias O_NONBLOCK = 0o4000
"""When opening a FIFO with O_RDONLY or O_WRONLY set:

- If O_NONBLOCK is set, an open() for reading-only shall return without delay.
    An open() for writing-only shall return an error if no process currently has
    the file open for reading.
- If O_NONBLOCK is clear, an open() for reading-only shall block the calling
    thread until a thread opens the file for writing. An open() for writing-only
    shall block the calling thread until a thread opens the file for reading.

When opening a block special or character special file that supports
non-blocking opens:

- If O_NONBLOCK is set, the open() function shall return without blocking for
    the device to be ready or available. Subsequent behavior of the device is
    device-specific.
- If O_NONBLOCK is clear, the open() function shall block the calling thread
    until the device is ready or available before returning.

Otherwise, the O_NONBLOCK flag shall not cause an error, but it is unspecified
whether the file status flags will include the O_NONBLOCK flag.
"""
alias O_CLOEXEC = 0o2000000
"""Atomically set the FD_CLOEXEC flag on the new file descriptor."""
alias O_DIRECTORY = 0o200000
"""Constant: O_DIRECTORY."""
alias O_DSYNC = 0o10000
"""Constant: O_DSYNC."""
alias O_NOCTTY = 0o400
"""Do not assign controlling terminal."""
alias O_NOFOLLOW = 0o400000
"""Do not follow symbolic links."""

alias F_DUPFD = 0
"""Constant: F_DUPFD."""
alias F_GETFD = 1
"""Constant: F_GETFD."""
alias F_SETFD = 2
"""Constant: F_SETFD."""
alias F_GETFL = 3
"""Constant: F_GETFL."""
alias F_SETFL = 4
"""Constant: F_SETFL."""
alias F_GETLK = 5
"""Constant: F_GETLK."""
alias F_SETLK = 6
"""Constant: F_SETLK."""
alias F_SETLKW = 7
"""Constant: F_SETLKW."""
alias F_SETOWN = 8
"""Constant: F_SETOWN."""
alias F_GETOWN = 9
"""Constant: F_GETOWN."""
alias F_RGETLK = 10
"""Constant: F_RGETLK."""
alias F_SETSIG = 10
"""Constant: F_SETSIG."""
alias F_RSETLK = 11
"""Constant: F_RSETLK."""
alias F_GETSIG = 11
"""Constant: F_GETSIG."""
alias F_CNVT = 12
"""Constant: F_CNVT."""
alias F_RSETLKW = 13
"""Constant: F_RSETLKW."""
alias F_INPROGRESS = 16
"""Constant: F_INPROGRESS."""
alias F_DUPFD_CLOEXEC = 1030
"""Constant: F_DUPFD_CLOEXEC."""
alias F_LINUX_SPECIFIC_BASE = 1024
"""Constant: F_LINUX_SPECIFIC_BASE."""


alias LOCK_SH = 1
"""Shared lock."""
alias LOCK_EX = 2
"""Exclusive lock."""
alias LOCK_NB = 4
"""Or'd with one of the above to prevent blocking."""
alias LOCK_UN = 8
"""Remove lock."""
alias LOCK_MAND = 32
"""This is a mandatory flock."""
alias LOCK_READ = 64
"""Which allows concurrent read operations."""
alias LOCK_WRITE = 128
"""Which allows concurrent write operations."""
alias LOCK_RW = 192
"""Which allows concurrent read & write ops."""


alias AT_EACCESS = 512
"""Constant: AT_EACCESS."""
alias AT_FDCWD = -100
"""Constant: AT_FDCWD."""
alias AT_SYMLINK_NOFOLLOW = 256
"""Constant: AT_SYMLINK_NOFOLLOW."""
alias AT_REMOVEDIR = 512
"""Constant: AT_REMOVEDIR."""
alias AT_SYMLINK_FOLLOW = 1024
"""Constant: AT_SYMLINK_FOLLOW."""
alias AT_NO_AUTOMOUNT = 2048
"""Constant: AT_NO_AUTOMOUNT."""
alias AT_EMPTY_PATH = 4096
"""Constant: AT_EMPTY_PATH."""
alias AT_RECURSIVE = 32768
"""Constant: AT_RECURSIVE."""

# ===----------------------------------------------------------------------=== #
# File modes (sys/stat.h)
# ===----------------------------------------------------------------------=== #

alias S_IRWXU = 0o700
"""Read, write, execute/search by owner."""
alias S_IRUSR = 0o400
"""Read permission, owner."""
alias S_IWUSR = 0o200
"""Write permission, owner."""
alias S_IXUSR = 0o100
"""Execute/search permission, owner."""
alias S_IRWXG = 0o70
"""Read, write, execute/search by group."""
alias S_IRGRP = 0o40
"""Read permission, group."""
alias S_IWGRP = 0o20
"""Write permission, group."""
alias S_IXGRP = 0o10
"""Execute/search permission, group."""
alias S_IRWXO = 0o7
"""Read, write, execute/search by others."""
alias S_IROTH = 0o4
"""Read permission, others."""
alias S_IWOTH = 0o2
"""Write permission, others."""
alias S_IXOTH = 0o1
"""Execute/search permission, others."""
alias S_ISUID = 0o4000
"""Set-user-ID on execution."""
alias S_ISGID = 0o2000
"""Set-group-ID on execution."""
alias S_ISVTX = 0o1000
"""On directories, restricted deletion flag."""

# ===----------------------------------------------------------------------=== #
# Logging constants (syslog.h)
# ===----------------------------------------------------------------------=== #

# levels
alias LOG_EMERG = 0
"""A panic condition was reported to all processes."""
alias LOG_ALERT = 1
"""A condition that should be corrected immediately."""
alias LOG_CRIT = 2
"""A critical condition."""
alias LOG_ERR = 3
"""An error message."""
alias LOG_WARNING = 4
"""A warning message."""
alias LOG_NOTICE = 5
"""A condition requiring special handling."""
alias LOG_INFO = 6
"""A general information message."""
alias LOG_DEBUG = 7
"""A message useful for debugging programs."""

# options
alias LOG_PID = 1
"""Log the process ID with each message."""
alias LOG_CONS = 2
"""Log to the system console on error."""
alias LOG_ODELAY = 4
"""Delay open until syslog() is called."""
alias LOG_NDELAY = 8
"""Connect to syslog daemon immediately."""
alias LOG_NOWAIT = 0x10
"""Do not wait for child processes."""
alias LOG_PERROR = 0x20
"""Log to stderr as well."""

# facilities
alias LOG_KERN = (0 << 3)
"""Kernel messages."""
alias LOG_USER = (1 << 3)
"""Random user-level messages."""
alias LOG_MAIL = (2 << 3)
"""Mail system."""
alias LOG_DAEMON = (3 << 3)
"""System daemons."""
alias LOG_AUTH = (4 << 3)
"""Security/authorization messages."""
alias LOG_SYSLOG = (5 << 3)
"""Messages generated internally by syslogd."""
alias LOG_LPR = (6 << 3)
"""Line printer subsystem."""
alias LOG_NEWS = (7 << 3)
"""Network news subsystem."""
alias LOG_UUCP = (8 << 3)
"""UUCP subsystem."""
alias LOG_CRON = (9 << 3)
"""Clock daemon."""
alias LOG_AUTHPRIV = (10 << 3)
"""Security/authorization messages (private)."""
alias LOG_FTP = (11 << 3)
"""Ftp daemon."""

alias LOG_LOCAL0 = (16 << 3)
"""Reserved for local use."""
alias LOG_LOCAL1 = (17 << 3)
"""Reserved for local use."""
alias LOG_LOCAL2 = (18 << 3)
"""Reserved for local use."""
alias LOG_LOCAL3 = (19 << 3)
"""Reserved for local use."""
alias LOG_LOCAL4 = (20 << 3)
"""Reserved for local use."""
alias LOG_LOCAL5 = (21 << 3)
"""Reserved for local use."""
alias LOG_LOCAL6 = (22 << 3)
"""Reserved for local use."""
alias LOG_LOCAL7 = (23 << 3)
"""Reserved for local use."""

alias LOG_NFACILITIES = 24
"""Current number of facilities."""
alias LOG_FACMASK = 0x03F8
"""Mask to extract facility part."""
