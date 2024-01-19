"""Provides the repository macro to import Triton."""
load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "5797d1b7c752ffe40538c902e1d929047bdc2127"
    TRITON_SHA256 = "2be1e9b12e585f5701794e1247f8f7b4cfe41251cc11b6b0c1b515c5c03c1af4"

    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-{commit}".format(commit = TRITON_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/triton/archive/{commit}.tar.gz".format(commit = TRITON_COMMIT)),
        # For temporary changes which haven't landed upstream yet.
        patch_file = [
            "//third_party/triton:b304456327.patch",
        ],
    )
