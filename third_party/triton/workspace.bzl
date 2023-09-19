"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "9f1a5f4"
    TRITON_SHA256 = "9f1a5f42df4813da438ad79b41b33eb685d54adf"

    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-{commit}".format(commit = TRITON_COMMIT),
        urls = tf_mirror_urls("https://github.com/ROCmSoftwarePlatform/triton/archive/{commit}.tar.gz".format(commit = TRITON_COMMIT)),
        # For temporary changes which haven't landed upstream yet.
        patch_file = [
            "//third_party/triton:cl536931041.patch",
            "//third_party/triton:cl555471166.patch",
            "//third_party/triton:cl561859552.patch",
            "//third_party/triton:msvc_fixes.patch",
        ],
    )
