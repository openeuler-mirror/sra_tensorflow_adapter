"""Starlark macros for ktfop.
"""

def if_ktfop(if_true, if_false = []):
    return select({
        "@org_tensorflow//third_party/ktfop:build_with_ktfop": if_true,
        "//conditions:default": if_false,
    })
