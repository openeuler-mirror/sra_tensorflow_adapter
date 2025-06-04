"""Provides the repository macro to import ktfop."""

def repo():
    """Imports ktfop."""

    native.new_local_repository(
        name = "ktfop_archive",
        build_file = "@org_tensorflow//third_party/ktfop:ktfop.BUILD",
        path = "/usr/local/sra_inference",
    )
