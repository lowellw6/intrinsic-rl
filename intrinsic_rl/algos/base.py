
from abc import ABC, abstractmethod

from rlpyt.algos.base import RlAlgorithm


class IntrinsicRlAlgorithm(RlAlgorithm, ABC):
    """
    Abstract class extending RlAlgorithm to support interface
    with intrinsic bonus model.
    """

    @abstractmethod
    def extract_bonus_inputs(self, *args, **kwargs):
        """Should extract and format all necessary inputs to the bonus model."""
        pass
