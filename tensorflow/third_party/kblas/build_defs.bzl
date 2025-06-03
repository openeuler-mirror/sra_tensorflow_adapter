"""Starlark macros for kblas.
"""

def if_kblas(if_true, if_false = []):
    return select({
        "@org_tensorflow//third_party/kblas:build_with_kblas": if_true,
        "//conditions:default": if_false,
    })
