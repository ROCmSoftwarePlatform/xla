"""Provides the repository macro to import Triton."""
load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "c178e087c7bc0bbb6b0241114b20e17c3fa1c639"
    TRITON_SHA256 = "a27bec068f773efbd8e1a381dae9d5938ec703405c442ea5c69867653cc7db90"
    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-{commit}".format(commit = TRITON_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/triton/archive/{commit}.tar.gz".format(commit = TRITON_COMMIT)),
        # For temporary changes which haven't landed upstream yet.
        patch_file = [],
    )
