"""Provides the repository macro to import kblas."""

def repo():
    """Imports kblas."""

    native.new_local_repository(
        name = "kblas_archive",
        build_file = "@org_tensorflow//third_party/kblas:kblas.BUILD",
        path = "/usr/local/kml",
    )
