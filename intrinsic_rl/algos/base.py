
from abc import ABC

from rlpyt.algos.base import RlAlgorithm


class IntrinsicRlAlgorithm(RlAlgorithm, ABC):
    """
    Abstract class extending RlAlgorithm to support interface
    with intrinsic bonus model.
    """

    pass
