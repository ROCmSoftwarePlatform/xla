"""Provides the repository macro to import Triton."""
load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "fde44d0cb2cc554fb7761c0915066569fede1b8b"
    TRITON_SHA256 = "1d24b1dcc01a9688ab14cbdb8c6b2d09c6bba2187ac7ccf4a126240154165617"

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
