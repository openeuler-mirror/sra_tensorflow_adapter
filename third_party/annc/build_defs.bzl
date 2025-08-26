def if_enable_annc(if_true, if_false = []):
    """Shorthand to select() if we are building with ANNC and ANNC is enabled.

    This is only effective when built with ANNC.

    Args:
        if_true: expression to evaluate if building with ANNC and ANNC is enabled      
        if_false: expression to evaluate if building without ANNC or ANNC is not enabled.

    Returns:
        A select evaluating to either if_true or if_false as appropriate.
    """
    return select({
        "@org_tensorflow//tensorflow:linux_aarch64": if_true,
        "//conditions:default": if_false,
    })