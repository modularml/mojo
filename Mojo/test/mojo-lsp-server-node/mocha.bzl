"Example macro wrapping the mocha CLI"

load("@cfg_workaround.bzl", "TARGET_CONSTRAINTS")
load("@mojo-lsp-server-node-tests//Mojo/test/mojo-lsp-server-node:mocha/package_json.bzl", "bin")
load("//bazel:api.bzl", "mojo_test_environment")

def mocha_test(name, srcs, args = [], data = [], env = {}, **kwargs):
    mojo_test_environment(name = "mojo_test_env", data = [
        "@mojo//:std",
        "//max:extensibility",
    ], testonly = True)

    bin.mocha_test(
        name = name,
        # The js_binary launcher embeds the Node.js binary path at analysis time
        # based on the exec platform. Under --config=remote-macos,
        # --extra_execution_platforms lists //:m7g-platform (linux) first, so
        # most actions — including js_binary builds — run on the linux executor,
        # embedding a linux node path that fails when the script runs on macOS.
        # Constrain the exec platform to TARGET_CONSTRAINTS (from
        # @cfg_workaround.bzl) so Bazel picks the macOS exec platform for this
        # target, which resolves the correct Node.js toolchain for that OS.
        exec_compatible_with = TARGET_CONSTRAINTS,
        args = [
            "--parallel",
            # NodeJS has to be explicitly told to enable source map support.
            "--node-option enable-source-maps",
            "--reporter",
            native.package_name() + "/src/reporter.js",
            "--config=$(location //Mojo/test/mojo-lsp-server-node:.mocharc.json)",
            native.package_name() + "/src/**/*.spec.js",
        ] + args,
        data = data + srcs + [
            "//Mojo/test/mojo-lsp-server-node:.mocharc.json",
            "@mojo//:std",
            "//max:extensibility",
            ":mojo_test_env",
        ],
        env = env | {
            # Add environment variable so that mocha writes its test xml
            # to the location Bazel expects.
            "MOCHA_FILE": "$$XML_OUTPUT_FILE",
            "MODULAR_MOJO_MAX_IMPORT_PATH": "$(COMPUTED_IMPORT_PATH)",
            "MODULAR_MOJO_MAX_SHARED_LIBS": "$(COMPUTED_LIBS)",
        },
        env_inherit = ["FORCE_COLOR"],
        toolchains = [":mojo_test_env"],
        **kwargs
    )
