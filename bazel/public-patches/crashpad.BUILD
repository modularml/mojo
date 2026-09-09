load("@build_bazel_apple_support//rules:apple_genrule.bzl", "apple_genrule")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:objc_library.bzl", "objc_library")
load("@rules_python//python:defs.bzl", "py_binary")

_MINI_CHROMIUM_SRCS = [
    "third_party/mini_chromium/mini_chromium/base/atomicops.h",
    "third_party/mini_chromium/mini_chromium/base/atomicops_internals_atomicword_compat.h",
    "third_party/mini_chromium/mini_chromium/base/atomicops_internals_portable.h",
    "third_party/mini_chromium/mini_chromium/base/auto_reset.h",
    "third_party/mini_chromium/mini_chromium/base/bit_cast.h",
    "third_party/mini_chromium/mini_chromium/base/check.h",
    "third_party/mini_chromium/mini_chromium/base/check_op.h",
    "third_party/mini_chromium/mini_chromium/base/compiler_specific.h",
    "third_party/mini_chromium/mini_chromium/base/cxx17_backports.h",
    "third_party/mini_chromium/mini_chromium/base/debug/alias.cc",
    "third_party/mini_chromium/mini_chromium/base/debug/alias.h",
    "third_party/mini_chromium/mini_chromium/base/files/file_path.cc",
    "third_party/mini_chromium/mini_chromium/base/files/file_path.h",
    "third_party/mini_chromium/mini_chromium/base/files/file_util.h",
    "third_party/mini_chromium/mini_chromium/base/files/file_util_posix.cc",
    "third_party/mini_chromium/mini_chromium/base/files/scoped_file.cc",
    "third_party/mini_chromium/mini_chromium/base/files/scoped_file.h",
    "third_party/mini_chromium/mini_chromium/base/format_macros.h",
    "third_party/mini_chromium/mini_chromium/base/logging.cc",
    "third_party/mini_chromium/mini_chromium/base/logging.h",
    "third_party/mini_chromium/mini_chromium/base/memory/free_deleter.h",
    "third_party/mini_chromium/mini_chromium/base/memory/page_size.h",
    "third_party/mini_chromium/mini_chromium/base/memory/page_size_posix.cc",
    "third_party/mini_chromium/mini_chromium/base/memory/scoped_policy.h",
    "third_party/mini_chromium/mini_chromium/base/metrics/histogram_functions.h",
    "third_party/mini_chromium/mini_chromium/base/metrics/histogram_macros.h",
    "third_party/mini_chromium/mini_chromium/base/metrics/persistent_histogram_allocator.h",
    "third_party/mini_chromium/mini_chromium/base/notreached.h",
    "third_party/mini_chromium/mini_chromium/base/numerics/checked_math.h",
    "third_party/mini_chromium/mini_chromium/base/numerics/checked_math_impl.h",
    "third_party/mini_chromium/mini_chromium/base/numerics/clamped_math.h",
    "third_party/mini_chromium/mini_chromium/base/numerics/clamped_math_impl.h",
    "third_party/mini_chromium/mini_chromium/base/numerics/safe_conversions.h",
    "third_party/mini_chromium/mini_chromium/base/numerics/safe_conversions_arm_impl.h",
    "third_party/mini_chromium/mini_chromium/base/numerics/safe_conversions_impl.h",
    "third_party/mini_chromium/mini_chromium/base/numerics/safe_math.h",
    "third_party/mini_chromium/mini_chromium/base/numerics/safe_math_arm_impl.h",
    "third_party/mini_chromium/mini_chromium/base/numerics/safe_math_clang_gcc_impl.h",
    "third_party/mini_chromium/mini_chromium/base/numerics/safe_math_shared_impl.h",
    "third_party/mini_chromium/mini_chromium/base/posix/eintr_wrapper.h",
    "third_party/mini_chromium/mini_chromium/base/posix/safe_strerror.cc",
    "third_party/mini_chromium/mini_chromium/base/posix/safe_strerror.h",
    "third_party/mini_chromium/mini_chromium/base/process/memory.cc",
    "third_party/mini_chromium/mini_chromium/base/process/memory.h",
    "third_party/mini_chromium/mini_chromium/base/rand_util.cc",
    "third_party/mini_chromium/mini_chromium/base/rand_util.h",
    "third_party/mini_chromium/mini_chromium/base/scoped_clear_last_error.h",
    "third_party/mini_chromium/mini_chromium/base/scoped_generic.h",
    "third_party/mini_chromium/mini_chromium/base/strings/string_number_conversions.cc",
    "third_party/mini_chromium/mini_chromium/base/strings/string_number_conversions.h",
    "third_party/mini_chromium/mini_chromium/base/strings/string_piece.h",
    "third_party/mini_chromium/mini_chromium/base/strings/string_util.cc",
    "third_party/mini_chromium/mini_chromium/base/strings/string_util.h",
    "third_party/mini_chromium/mini_chromium/base/strings/string_util_posix.h",
    "third_party/mini_chromium/mini_chromium/base/strings/stringprintf.cc",
    "third_party/mini_chromium/mini_chromium/base/strings/stringprintf.h",
    "third_party/mini_chromium/mini_chromium/base/strings/sys_string_conversions.h",
    "third_party/mini_chromium/mini_chromium/base/strings/utf_string_conversion_utils.cc",
    "third_party/mini_chromium/mini_chromium/base/strings/utf_string_conversion_utils.h",
    "third_party/mini_chromium/mini_chromium/base/strings/utf_string_conversions.cc",
    "third_party/mini_chromium/mini_chromium/base/strings/utf_string_conversions.h",
    "third_party/mini_chromium/mini_chromium/base/synchronization/condition_variable.h",
    "third_party/mini_chromium/mini_chromium/base/synchronization/condition_variable_posix.cc",
    "third_party/mini_chromium/mini_chromium/base/synchronization/lock.cc",
    "third_party/mini_chromium/mini_chromium/base/synchronization/lock.h",
    "third_party/mini_chromium/mini_chromium/base/synchronization/lock_impl.h",
    "third_party/mini_chromium/mini_chromium/base/synchronization/lock_impl_posix.cc",
    "third_party/mini_chromium/mini_chromium/base/sys_byteorder.h",
    "third_party/mini_chromium/mini_chromium/base/template_util.h",
    "third_party/mini_chromium/mini_chromium/base/third_party/icu/icu_utf.cc",
    "third_party/mini_chromium/mini_chromium/base/third_party/icu/icu_utf.h",
    "third_party/mini_chromium/mini_chromium/base/threading/thread_local_storage.cc",
    "third_party/mini_chromium/mini_chromium/base/threading/thread_local_storage.h",
    "third_party/mini_chromium/mini_chromium/base/threading/thread_local_storage_posix.cc",
    "third_party/mini_chromium/mini_chromium/build/build_config.h",
    "third_party/mini_chromium/mini_chromium/build/buildflag.h",
    "third_party/mini_chromium/mini_chromium/testing/platform_test.h",
]

objc_library(
    name = "mini_chromium_macos",
    srcs = _MINI_CHROMIUM_SRCS + [
        "third_party/mini_chromium/mini_chromium/base/mac/close_nocancel.cc",
        "third_party/mini_chromium/mini_chromium/base/mac/foundation_util.h",
        "third_party/mini_chromium/mini_chromium/base/mac/mach_logging.cc",
        "third_party/mini_chromium/mini_chromium/base/mac/mach_logging.h",
        "third_party/mini_chromium/mini_chromium/base/mac/scoped_cftyperef.h",
        "third_party/mini_chromium/mini_chromium/base/mac/scoped_ioobject.h",
        "third_party/mini_chromium/mini_chromium/base/mac/scoped_launch_data.h",
        "third_party/mini_chromium/mini_chromium/base/mac/scoped_mach_port.cc",
        "third_party/mini_chromium/mini_chromium/base/mac/scoped_mach_port.h",
        "third_party/mini_chromium/mini_chromium/base/mac/scoped_mach_vm.cc",
        "third_party/mini_chromium/mini_chromium/base/mac/scoped_mach_vm.h",
        "third_party/mini_chromium/mini_chromium/base/mac/scoped_nsautorelease_pool.h",
        "third_party/mini_chromium/mini_chromium/base/mac/scoped_nsobject.h",
        "third_party/mini_chromium/mini_chromium/base/mac/scoped_typeref.h",
    ],
    copts = ["-std=c++14"],
    includes = ["third_party/mini_chromium/mini_chromium"],
    linkopts = [
        "-Wl,-framework,ApplicationServices",
        "-Wl,-framework,CoreFoundation",
        "-Wl,-framework,Foundation",
        "-Wl,-framework,IOKit",
        "-Wl,-framework,Security",
    ],
    non_arc_srcs = [
        "third_party/mini_chromium/mini_chromium/base/mac/foundation_util.mm",
        "third_party/mini_chromium/mini_chromium/base/mac/scoped_nsautorelease_pool.mm",
        "third_party/mini_chromium/mini_chromium/base/strings/sys_string_conversions_mac.mm",
    ],
    target_compatible_with = ["@platforms//os:macos"],
)

cc_library(
    name = "mini_chromium_linux",
    srcs = _MINI_CHROMIUM_SRCS,
    copts = ["-std=c++14"],
    includes = ["third_party/mini_chromium/mini_chromium"],
    target_compatible_with = ["@platforms//os:linux"],
)

alias(
    name = "mini_chromium",
    actual = select({
        "@platforms//os:linux": ":mini_chromium_linux",
        "@platforms//os:macos": ":mini_chromium_macos",
    }),
)

cc_library(
    name = "backtrace_common",
    defines = [
        "CRASHPAD_FLOCK_ALWAYS_SUPPORTED=1",
        "CRASHPAD_ZLIB_SOURCE_EXTERNAL_WITH_EMBEDDED_BUILD",
        "MINI_CHROMIUM_INTERNAL_BUILDFLAG_VALUE_IS_CHROMEOS_ASH=MINI_CHROMIUM_INTERNAL_BUILDFLAG_VALUE_IS_CHROMEOS",
        "MINI_CHROMIUM_INTERNAL_BUILDFLAG_VALUE_IS_CHROMEOS_LACROS=MINI_CHROMIUM_INTERNAL_BUILDFLAG_VALUE_IS_CHROMEOS",
    ],
)

cc_library(
    name = "zlib",
    hdrs = ["third_party/zlib/zlib_crashpad.h"],
    defines = ["CRASHPAD_ZLIB_SOURCE_EXTERNAL_WITH_EMBEDDED_BUILD"],
    deps = ["@llvm_zlib//:zlib"],
)

cc_library(
    name = "lss",
    hdrs = glob(["third_party/lss/**/*.h"]),
    defines = ["CRASHPAD_LSS_SOURCE_EMBEDDED"],
)

_UTIL_MULTIPLATFORM_SRCS = [
    "util/backtrace/crash_loop_detection.cc",
    "util/file/delimited_file_reader.cc",
    "util/file/directory_reader_posix.cc",
    "util/file/file_helper.cc",
    "util/file/file_io.cc",
    "util/file/file_io_posix.cc",
    "util/file/file_reader.cc",
    "util/file/file_seeker.cc",
    "util/file/file_writer.cc",
    "util/file/filesystem_posix.cc",
    "util/file/output_stream_file_writer.cc",
    "util/file/scoped_remove_file.cc",
    "util/file/string_file.cc",
    "util/misc/clock_posix.cc",
    "util/misc/initialization_state_dcheck.cc",
    "util/misc/lexing.cc",
    "util/misc/metrics.cc",
    "util/misc/pdb_structures.cc",
    "util/misc/random_string.cc",
    "util/misc/range_set.cc",
    "util/misc/reinterpret_bytes.cc",
    "util/misc/scoped_forbid_return.cc",
    "util/misc/time.cc",
    "util/misc/uuid.cc",
    "util/misc/zlib.cc",
    "util/net/http_body.cc",
    "util/net/http_body_gzip.cc",
    "util/net/http_multipart_builder.cc",
    "util/net/http_transport.cc",
    "util/net/url.cc",
    "util/numeric/checked_address_range.cc",
    "util/posix/close_multiple.cc",
    "util/posix/close_stdio.cc",
    "util/posix/scoped_dir.cc",
    "util/posix/signals.cc",
    "util/posix/spawn_subprocess.cc",
    "util/process/process_memory.cc",
    "util/process/process_memory_range.cc",
    "util/process/process_memory_sanitized.cc",
    "util/stdlib/aligned_allocator.cc",
    "util/stdlib/string_number_conversion.cc",
    "util/stdlib/strlcpy.cc",
    "util/stdlib/strnlen.cc",
    "util/stream/base94_output_stream.cc",
    "util/stream/file_encoder.cc",
    "util/stream/file_output_stream.cc",
    "util/stream/log_output_stream.cc",
    "util/stream/test_output_stream.cc",
    "util/stream/zlib_output_stream.cc",
    "util/string/split_string.cc",
    "util/synchronization/semaphore_posix.cc",
    "util/thread/thread.cc",
    "util/thread/thread_log_messages.cc",
    "util/thread/thread_posix.cc",
    "util/thread/worker_thread.cc",
]

cc_library(
    name = "util_linux",
    srcs = _UTIL_MULTIPLATFORM_SRCS + [
        "package.h",
        "util/linux/auxiliary_vector.cc",
        "util/linux/direct_ptrace_connection.cc",
        "util/linux/exception_handler_client.cc",
        "util/linux/exception_handler_protocol.cc",
        "util/linux/memory_map.cc",
        "util/linux/proc_stat_reader.cc",
        "util/linux/proc_task_reader.cc",
        "util/linux/ptrace_broker.cc",
        "util/linux/ptrace_client.cc",
        "util/linux/ptracer.cc",
        "util/linux/scoped_pr_set_dumpable.cc",
        "util/linux/scoped_pr_set_ptracer.cc",
        "util/linux/scoped_ptrace_attach.cc",
        "util/linux/socket.cc",
        "util/linux/thread_info.cc",
        "util/misc/capture_context_linux.S",
        "util/misc/paths_linux.cc",
        "util/misc/time_linux.cc",
        "util/net/http_transport_libcurl.cc",
        "util/posix/process_info_linux.cc",
        "util/posix/scoped_mmap.cc",
        "util/process/process_memory_linux.cc",
    ],
    hdrs = glob(["util/**/*.h"]),
    copts = [
        "-std=c++17",
        "-Wno-nontrivial-memcall",
        "-fno-sanitize=undefined",  # Known issues
    ],
    target_compatible_with = ["@platforms//os:linux"],
    textual_hdrs = ["util/misc/arm64_pac_bti.S"],
    deps = [
        ":backtrace_common",
        ":lss",
        ":mini_chromium",
        ":zlib",
        "@curl",
    ],
)

py_binary(
    name = "mig",
    srcs = [
        "util/mach/mig.py",
        "util/mach/mig_fix.py",
        "util/mach/mig_gen.py",
    ],
    imports = ["util/mach"],
)

apple_genrule(
    name = "mach_gen_child_port",
    srcs = ["util/mach/child_port.defs"],
    outs = [
        "child_port.h",
        "child_portServer.c",
        "child_portServer.h",
        "child_portUser.c",
    ],
    cmd = "mkdir -p $$(dirname $(location :{name}User.c)) && $(location :mig) --arch arm64 --sdk $$SDKROOT $(SRCS) $(location :{name}User.c) $(location :{name}Server.c) $(location :{name}.h) $(location :{name}Server.h)".format(
        name = "child_port",
    ),
    target_compatible_with = ["@platforms//os:macos"],
    tools = [":mig"],
)

_MIG_SRCS = [
    "exc",
    "mach_exc",
    "notify",
]

[
    apple_genrule(
        name = "mach_gen_" + name,
        outs = [
            name + ext
            for ext in [
                "User.c",
                "Server.c",
                ".h",
                "Server.h",
            ]
        ],
        cmd = "mkdir -p $$(dirname $(location :{name}User.c)) && $(location :mig) --arch arm64 --sdk $$SDKROOT $$SDKROOT/usr/include/mach/{name}.defs $(location :{name}User.c) $(location :{name}Server.c) $(location :{name}.h) $(location :{name}Server.h)".format(
            name = name,
        ),
        target_compatible_with = ["@platforms//os:macos"],
        tools = [":mig"],
    )
    for name in _MIG_SRCS
]

cc_library(
    name = "all_mig_srcs",
    srcs = [
        "util/mach/child_port_types.h",
        ":mach_gen_child_port",
    ] + [
        ":mach_gen_" + name
        for name in _MIG_SRCS
    ],
    hdrs = [
        name + ext
        for name in _MIG_SRCS + [
            "child_port",
        ]
        for ext in [
            ".h",
            "Server.h",
        ]
    ],
    include_prefix = "util/mach",
    target_compatible_with = ["@platforms//os:macos"],
)

objc_library(
    name = "util_macos",
    srcs = _UTIL_MULTIPLATFORM_SRCS + [
        "package.h",
        "util/mac/mac_util.cc",
        "util/mac/service_management.cc",
        "util/mac/sysctl.cc",
        "util/mac/xattr.cc",
        "util/mach/bootstrap.cc",
        "util/mach/child_port_handshake.cc",
        "util/mach/child_port_server.cc",
        "util/mach/composite_mach_message_server.cc",
        "util/mach/exc_client_variants.cc",
        "util/mach/exc_server_variants.cc",
        "util/mach/exception_behaviors.cc",
        "util/mach/exception_ports.cc",
        "util/mach/exception_types.cc",
        "util/mach/mach_extensions.cc",
        "util/mach/mach_message.cc",
        "util/mach/mach_message_server.cc",
        "util/mach/notify_server.cc",
        "util/mach/scoped_task_suspend.cc",
        "util/mach/symbolic_constants_mach.cc",
        "util/misc/capture_context_mac.S",
        "util/misc/clock_mac.cc",
        "util/misc/paths_mac.cc",
        "util/posix/process_info_mac.cc",
        "util/process/process_memory_mac.cc",
        "util/synchronization/semaphore_mac.cc",
    ],
    hdrs = glob(["util/**/*.h"]),
    copts = [
        "-Wno-nontrivial-memcall",
        "-Wno-non-virtual-dtor",
        "-Wno-suggest-override",
        "-fno-sanitize=undefined",  # Known issues
    ],
    cxxopts = [
        "-std=c++17",
    ],
    linkopts = ["-lbsm"],
    non_arc_srcs = [
        "util/net/http_transport_mac.mm",
    ],
    deps = [
        ":all_mig_srcs",
        ":backtrace_common",
        ":compat",
        ":lss",
        ":mini_chromium_macos",
        ":zlib",
    ],
)

alias(
    name = "util",
    actual = select({
        "@platforms//os:linux": ":util_linux",
        "@platforms//os:macos": ":util_macos",
    }),
)

cc_library(
    name = "compat",
    hdrs = glob(["compat/**/*.h"]),
    includes = ["compat/non_win"] + select({
        "@platforms//os:linux": ["compat/linux"],
        "@platforms//os:macos": [],
    }),
)

_CLIENT_SRCS = [
    "client/annotation.cc",
    "client/annotation_list.cc",
    "client/crash_report_database.cc",
    "client/crashpad_info.cc",
    "client/prune_crash_reports.cc",
    "client/settings.cc",
]

objc_library(
    name = "client_macos",
    srcs = _CLIENT_SRCS + [
        "client/crashpad_client_mac.cc",
        "client/simulate_crash_mac.cc",
    ],
    hdrs = glob(["client/**/*.h"]),
    copts = [
        "-Wno-suggest-override",
        "-Wno-non-virtual-dtor",
        "-fno-sanitize=undefined",  # Known issues
    ],
    non_arc_srcs = [
        "client/crash_report_database_mac.mm",
    ],
    target_compatible_with = ["@platforms//os:macos"],
    deps = [
        ":compat",
        ":mini_chromium",
        ":util",
    ],
)

cc_library(
    name = "client_linux",
    srcs = _CLIENT_SRCS + [
        "client/client_argv_handling.cc",
        "client/crash_report_database_generic.cc",
        "client/crashpad_client_linux.cc",
        "client/crashpad_info_note.S",
        "client/pthread_create_linux.cc",
    ],
    hdrs = glob(["client/**/*.h"]),
    copts = [
        "-fno-sanitize=undefined",  # Known issues
    ],
    target_compatible_with = ["@platforms//os:linux"],
    deps = [
        ":compat",
        ":mini_chromium",
        ":util",
    ],
)

alias(
    name = "client",
    actual = select({
        "@platforms//os:linux": ":client_linux",
        "@platforms//os:macos": ":client_macos",
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "minidump",
    srcs = [
        "minidump/minidump_annotation_writer.cc",
        "minidump/minidump_annotation_writer.h",
        "minidump/minidump_byte_array_writer.cc",
        "minidump/minidump_byte_array_writer.h",
        "minidump/minidump_context.h",
        "minidump/minidump_context_writer.cc",
        "minidump/minidump_context_writer.h",
        "minidump/minidump_crashpad_info_writer.cc",
        "minidump/minidump_crashpad_info_writer.h",
        "minidump/minidump_exception_writer.cc",
        "minidump/minidump_exception_writer.h",
        "minidump/minidump_extensions.cc",
        "minidump/minidump_extensions.h",
        "minidump/minidump_file_writer.cc",
        "minidump/minidump_file_writer.h",
        "minidump/minidump_handle_writer.cc",
        "minidump/minidump_handle_writer.h",
        "minidump/minidump_memory_info_writer.cc",
        "minidump/minidump_memory_info_writer.h",
        "minidump/minidump_memory_writer.cc",
        "minidump/minidump_memory_writer.h",
        "minidump/minidump_misc_info_writer.cc",
        "minidump/minidump_misc_info_writer.h",
        "minidump/minidump_module_crashpad_info_writer.cc",
        "minidump/minidump_module_crashpad_info_writer.h",
        "minidump/minidump_module_writer.cc",
        "minidump/minidump_module_writer.h",
        "minidump/minidump_rva_list_writer.cc",
        "minidump/minidump_rva_list_writer.h",
        "minidump/minidump_simple_string_dictionary_writer.cc",
        "minidump/minidump_simple_string_dictionary_writer.h",
        "minidump/minidump_stream_writer.cc",
        "minidump/minidump_stream_writer.h",
        "minidump/minidump_string_writer.cc",
        "minidump/minidump_string_writer.h",
        "minidump/minidump_system_info_writer.cc",
        "minidump/minidump_system_info_writer.h",
        "minidump/minidump_thread_id_map.cc",
        "minidump/minidump_thread_id_map.h",
        "minidump/minidump_thread_name_list_writer.cc",
        "minidump/minidump_thread_name_list_writer.h",
        "minidump/minidump_thread_writer.cc",
        "minidump/minidump_thread_writer.h",
        "minidump/minidump_unloaded_module_writer.cc",
        "minidump/minidump_unloaded_module_writer.h",
        "minidump/minidump_user_extension_stream_data_source.cc",
        "minidump/minidump_user_extension_stream_data_source.h",
        "minidump/minidump_user_stream_writer.cc",
        "minidump/minidump_user_stream_writer.h",
        "minidump/minidump_writable.cc",
        "minidump/minidump_writable.h",
        "minidump/minidump_writer_util.cc",
        "minidump/minidump_writer_util.h",
        "package.h",
    ],
    copts = [
        "-std=c++17",
    ],
    deps = [
        ":compat",
        ":mini_chromium",
        ":snapshot.headers",
        ":util",
    ],
)

cc_library(
    name = "snapshot.headers",
    hdrs = glob(["snapshot/**/*.h"]),
)

cc_library(
    name = "snapshot",
    srcs = [
        "snapshot/annotation_snapshot.cc",
        "snapshot/capture_memory.cc",
        "snapshot/cpu_context.cc",
        "snapshot/crashpad_info_client_options.cc",
        "snapshot/crashpad_types/image_annotation_reader.cc",
        "snapshot/handle_snapshot.cc",
        "snapshot/memory_snapshot.cc",
        "snapshot/minidump/exception_snapshot_minidump.cc",
        "snapshot/minidump/memory_snapshot_minidump.cc",
        "snapshot/minidump/minidump_annotation_reader.cc",
        "snapshot/minidump/minidump_context_converter.cc",
        "snapshot/minidump/minidump_simple_string_dictionary_reader.cc",
        "snapshot/minidump/minidump_string_list_reader.cc",
        "snapshot/minidump/minidump_string_reader.cc",
        "snapshot/minidump/module_snapshot_minidump.cc",
        "snapshot/minidump/process_snapshot_minidump.cc",
        "snapshot/minidump/system_snapshot_minidump.cc",
        "snapshot/minidump/thread_snapshot_minidump.cc",
        "snapshot/posix/timezone.cc",
        "snapshot/sanitized/memory_snapshot_sanitized.cc",
        "snapshot/sanitized/module_snapshot_sanitized.cc",
        "snapshot/sanitized/process_snapshot_sanitized.cc",
        "snapshot/sanitized/sanitization_information.cc",
        "snapshot/sanitized/thread_snapshot_sanitized.cc",
        "snapshot/unloaded_module_snapshot.cc",
        "snapshot/x86/cpuid_reader.cc",
    ] + select({
        "@platforms//os:linux": [
            "snapshot/crashpad_types/crashpad_info_reader.cc",
            "snapshot/elf/elf_dynamic_array_reader.cc",
            "snapshot/elf/elf_image_reader.cc",
            "snapshot/elf/elf_symbol_table_reader.cc",
            "snapshot/elf/module_snapshot_elf.cc",
            "snapshot/linux/capture_memory_delegate_linux.cc",
            "snapshot/linux/cpu_context_linux.cc",
            "snapshot/linux/debug_rendezvous.cc",
            "snapshot/linux/exception_snapshot_linux.cc",
            "snapshot/linux/process_reader_linux.cc",
            "snapshot/linux/process_snapshot_linux.cc",
            "snapshot/linux/system_snapshot_linux.cc",
            "snapshot/linux/thread_snapshot_linux.cc",
        ],
        "@platforms//os:macos": [
            "snapshot/mac/cpu_context_mac.cc",
            "snapshot/mac/exception_snapshot_mac.cc",
            "snapshot/mac/mach_o_image_annotations_reader.cc",
            "snapshot/mac/mach_o_image_reader.cc",
            "snapshot/mac/mach_o_image_segment_reader.cc",
            "snapshot/mac/mach_o_image_symbol_table_reader.cc",
            "snapshot/mac/module_snapshot_mac.cc",
            "snapshot/mac/process_reader_mac.cc",
            "snapshot/mac/process_snapshot_mac.cc",
            "snapshot/mac/process_types.cc",
            "snapshot/mac/process_types/custom.cc",
            "snapshot/mac/system_snapshot_mac.cc",
            "snapshot/mac/thread_snapshot_mac.cc",
        ],
    }),
    hdrs = glob(["snapshot/**/*.h"]),
    copts = [
        "-std=c++17",
        "-Wno-suggest-override",
    ],
    textual_hdrs = glob(["snapshot/mac/process_types/*.proctype"]),
    deps = [
        ":client",
        ":compat",
        ":mini_chromium",
        ":minidump",
        ":util",
    ],
)

cc_library(
    name = "crashpad_tools",
    srcs = [
        "tools/tool_support.cc",
        "tools/tool_support.h",
    ],
    copts = ["-std=c++14"],
    deps = [
        ":mini_chromium",
        ":util",
    ],
)

cc_binary(
    name = "modular-crashpad-handler",
    srcs = [
        "handler/crash_report_upload_thread.cc",
        "handler/crash_report_upload_thread.h",
        "handler/handler_main.cc",
        "handler/handler_main.h",
        "handler/main.cc",
        "handler/minidump_to_upload_parameters.cc",
        "handler/minidump_to_upload_parameters.h",
        "handler/prune_crash_reports_thread.cc",
        "handler/prune_crash_reports_thread.h",
        "handler/user_stream_data_source.cc",
        "handler/user_stream_data_source.h",
    ] + select({
        "@platforms//os:linux": [
            "handler/linux/capture_snapshot.cc",
            "handler/linux/capture_snapshot.h",
            "handler/linux/crash_report_exception_handler.cc",
            "handler/linux/crash_report_exception_handler.h",
            "handler/linux/exception_handler_server.cc",
            "handler/linux/exception_handler_server.h",
        ],
        "@platforms//os:macos": [
            "handler/mac/crash_report_exception_handler.cc",
            "handler/mac/crash_report_exception_handler.h",
            "handler/mac/exception_handler_server.cc",
            "handler/mac/exception_handler_server.h",
            "handler/mac/file_limit_annotation.cc",
            "handler/mac/file_limit_annotation.h",
        ],
    }),
    copts = [
        "-std=c++17",
        "-Wno-non-virtual-dtor",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":client",
        ":compat",
        ":crashpad_tools",
        ":minidump",
        ":snapshot",
    ],
)
