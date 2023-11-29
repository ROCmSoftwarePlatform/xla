"""Provides the repository macro to import Triton."""
load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "49f508a91a5f1eb859e8ef8c356937ce7739acaa"
    TRITON_SHA256 = "3cc7e2d82be39876f775f865971337a01ef02e1bd6894c6cc0994417750f3691"

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
