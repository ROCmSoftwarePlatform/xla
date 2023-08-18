"""Provides the repository macro to import Triton."""
load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Triton."""
    """TRITON_COMMIT = "95350db6c6e176b4e96d0c1ae0cbfb2ae965c785"""
    TRITON_COMMIT = "ee50dc71fb59ef248a69fbaee8ab8e72932d1645"
    """TRITON_SHA256 = "5b21151b0ef52d65ad76815256727b18a25ea0ff333118a7f4ab10683ff6a493"""
    TRITON_SHA256 = "b3ab533bed0a13b0996167caf85bc17a4dceed537edd0937dc48a2ac896a4a88"
    """TRITON_XLA_URL = "https://github.com/ROCmSoftwarePlatform/triton/archive/{commit}.tar.gz"""
    TRITON_XLA_URL = "https://github.com/zoranjovanovic-ns/triton/archive/{commit}.tar.gz"
    patch_file_ = [
            "//third_party/triton:cl536931041.patch",
            "//third_party/triton:cl550499635.patch",
            "//third_party/triton:cl551490193.patch",
    ]

    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-{commit}".format(commit = TRITON_COMMIT),
        urls = tf_mirror_urls(TRITON_XLA_URL.format(commit = TRITON_COMMIT)),
        patch_file = patch_file_
    )
