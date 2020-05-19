
import collections

def trimSeq(sequence):
    """Removes Python sequence container (e.g. tuple, list, etc.) if length one."""
    return sequence[0] if len(sequence) == 1 else sequence

def wrap(item):
    """Wraps item in tuple if not already a sequence."""
    if not isinstance(item, collections.Sequence):
        item = (item,)
    return item
