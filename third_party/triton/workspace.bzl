"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "f06f40bfe6cee7d48fb8da19ec845bfb1cb87696"
    TRITON_SHA256 = "add03cd7ea4fa8c26312b5d94d8804585b22c3b4f2dd3d52cdc118a3e945d877"

    #urls = tf_mirror_urls("https://github.com/openxla/triton/archive/{commit}.tar.gz".format(commit = TRITON_COMMIT)),
    TRITON_XLA_URL = "https://github.com/ROCmSoftwarePlatform/triton/archive/{commit}.tar.gz"
    # For temporary changes which haven't landed upstream yet.
    patch_file_ = [
    ]

    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-{commit}".format(commit = TRITON_COMMIT),
        urls = tf_mirror_urls(TRITON_XLA_URL.format(commit = TRITON_COMMIT)),
        patch_file = patch_file_
    )
