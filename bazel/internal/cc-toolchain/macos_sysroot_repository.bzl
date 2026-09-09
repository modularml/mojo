"""Create a local repository for the macOS SDK."""

# Based on https://cs.opensource.google/pigweed/pigweed/+/main:pw_toolchain/xcode.bzl

def _macos_sysroot_repository_impl(rctx):
    if rctx.os.name != "mac os x":
        # Write an empty file that will fail if used by otherwise allows analysis
        rctx.file("sysroot/BUILD.bazel", """\
load("@bazel_skylib//rules/directory:directory.bzl", "directory")

directory(
    name = "root",
    srcs = [],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "directory",
    srcs = [],
    visibility = ["//visibility:public"],
)
""")
        return

    developer_dir = rctx.getenv("DEVELOPER_DIR")
    result = rctx.execute(
        ["/usr/bin/xcrun", "--show-sdk-path", "--sdk", "macosx"],
        environment = {"DEVELOPER_DIR": developer_dir},
    )
    if result.return_code != 0:
        fail("Failed locating macOS SDK: {}".format(result.stderr))

    sdk_path = rctx.path(result.stdout.strip())
    for child in sdk_path.readdir(watch = "no"):
        rctx.symlink(child, "sysroot/" + child.basename)
    rctx.file("sysroot/BUILD.bazel", """\
load("@bazel_skylib//rules/directory:directory.bzl", "directory")

# NOTE: Ruby.framework has infinite symlink loops, so we have to avoid that in our globs
# NOTE: Including the smallest set of files is ideal to avoid more cache usage
_INCLUDES = [
    "usr/lib/**",
    "usr/include/**",
    "System/Library/Frameworks/ApplicationServices.framework/**",
    "System/Library/Frameworks/CFNetwork.framework/**",
    "System/Library/Frameworks/ColorSync.framework/**",
    "System/Library/Frameworks/CoreFoundation.framework/**",
    "System/Library/Frameworks/CoreGraphics.framework/**",
    "System/Library/Frameworks/CoreServices.framework/**",
    "System/Library/Frameworks/CoreText.framework/**",
    "System/Library/Frameworks/DiskArbitration.framework/**",
    "System/Library/Frameworks/ImageIO.framework/**",
    "System/Library/Frameworks/Foundation.framework/**",
    "System/Library/Frameworks/IOKit.framework/**",
    "System/Library/Frameworks/Metal.framework/**",
    "System/Library/Frameworks/Security.framework/**",
    "System/Library/Frameworks/SystemConfiguration.framework/**",
]

directory(
    name = "root",
    srcs = glob(_INCLUDES),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "directory",
    srcs = glob(_INCLUDES),
    visibility = ["//visibility:public"],
)
""")

macos_sysroot_repository = repository_rule(
    implementation = _macos_sysroot_repository_impl,
    environ = [
        "XCODE_VERSION",
        "DEVELOPER_DIR",
    ],
    local = True,
    configure = True,
)
