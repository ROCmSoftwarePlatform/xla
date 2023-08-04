"""Provides the repository macro to import Triton."""
load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Triton."""
    if "%{rocm_is_configured}" == "True":
        TRITON_COMMIT = "ifu-230620"
        TRITON_SHA256 = "f3e339e5f46736e31f117a10fcb3c6b7be02be91"
        TRITON_XLA_URL = "https://github.com/ROCmSoftwarePlatform/triton/archive/{commit}.tar.gz"
        patch_file_ = []
    else:
        TRITON_COMMIT = "cl546794996"
        TRITON_SHA256 = "57d4b5f1e68bb4df93528bd5394ba3338bef7bf9c0afdc96b44371fba650c037"
        TRITON_XLA_URL = "https://github.com/openxla/triton/archive/{commit}.tar.gz"
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
    