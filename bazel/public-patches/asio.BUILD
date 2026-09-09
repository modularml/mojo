load("@rules_cc//cc:cc_library.bzl", "cc_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "asio",
    hdrs = glob([
        "asio/include/**/*.hpp",
        "asio/include/**/*.ipp",
    ]),
    defines = [
        "ASIO_STANDALONE",
        "ASIO_HEADER_ONLY",
        "ASIO_NO_TYPEID",
    ],
    includes = ["asio/include"],
)
