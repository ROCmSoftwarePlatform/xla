"""Provides the repository macro to import Triton."""
load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "2a8fac9ba4cf712af5f12287ed362dcadf8049c7"
    TRITON_SHA256 = "7b13c1d4c83899e97d13b10dacd5bc8880960d12dd8baf791ee6a0f5ca6eaf94"

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
