
def trimSeq(sequence):
    """Removes Python sequence container (e.g. tuple, list, etc.) if length one."""
    return sequence[0] if len(sequence) == 1 else sequence
