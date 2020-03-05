
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallelCPU as DDPC

from rlpyt.agents.base import BaseAgent
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger


class IntrinsicBonusAgent(BaseAgent):
    """
    Augments agents with a second model which generates some intrinsic bonus.

    Note the bonus model is not expected to be used anywhere other than the optimizer process
    (and not in the sampler), so it is never placed in shared memory.
    """

    def __init__(self, *args, BonusModelCls=None, bonus_model_kwargs=None, initial_bonus_model_state_dict=None, **kwargs):
        """
        Intrinsic bonus model info saved but not initialized.

        :param BonusModelCls: The bonus model class to be used.
        :param bonus_model_kwargs: Any keyword arguments to pass when instantiating the bonus model.
        :param initial_bonus_model_state_dict: Initial bonus model parameter values.
        """

        save__init__args(locals())
        super().__init__(*args, **kwargs)
        self.bonus_model = None  # type: torch.nn.Module
        self.shared_bonus_model = None
        if self.bonus_model_kwargs is None:
            self.bonus_model_kwargs = dict()

    def initialize(self, env_spaces, share_memory=False, **kwargs):
        """
        Instantiate bonus model neural net model(s) according to environment interfaces.

        The bonus model is not expected to be used in sampler action-selection, so
        there is no need to put it in shared memory.

        Typically called in the sampler during startup.

        :param env_spaces: passed to ``make_env_to_bonus_model_kwargs()``, typically namedtuple of 'observation' and 'action'.
        :param share_memory: whether to use shared memory for bonus_model parameters.
        """
        super().initialize(env_spaces, share_memory=share_memory, **kwargs)
        self.env_bonus_model_kwargs = self.make_env_to_bonus_model_kwargs(env_spaces)
        self.bonus_model = self.BonusModelCls(**self.env_bonus_model_kwargs, **self.bonus_model_kwargs)
        if self.initial_bonus_model_state_dict is not None:
            self.bonus_model.load_state_dict(self.inital_bonus_model_state_dict)

    def make_env_to_bonus_model_kwargs(self, env_spaces):
        """Generate any keyword args to the bonus model which depend on environment interfaces."""
        return {}

    def to_device(self, cuda_idx=None):
        """
        Moves the bonus model to the specified cuda device, if not ``None``.

        Typically called in the runner during startup.
        """
        super().to_device(cuda_idx=cuda_idx)  # Sets self.device to cuda; not repeated here
        if cuda_idx is None:
            return
        self.bonus_model.to(self.device)
        logger.log(f"Initialized agent intrinsic bonus model on device: {self.device}")

    def data_parallel(self):
        """
        Wraps the intrinsic bonus model with PyTorch's DistributedDataParallel.  The
        intention is for rlpyt to create a separate Python process to drive
        each GPU (or CPU-group for CPU-only, MPI-like configuration).

        Typically called in the runner during startup.
        """
        super().data_parallel()
        if self.device.type == "cpu":
            self.bonus_model = DDPC(self.bonus_model)
            logger.log("Initialized DistributedDataParallelCPU intrinsic bonus model.")
        else:
            self.bonus_model = DDP(self.bonus_model,
                device_ids=[self.device.index], output_device=self.device.index)
            logger.log(f"Initialized DistributedDataParallel intrinsic bonus model on device {self.device}.")

    def parameters(self):
        """Overrides to share both base model and bonus model with optimizer."""
        return iter(list(self.model.parameters()) + list(self.bonus_model.parameters()))

    def state_dict(self):
        """Returns both base and bonus model parameters for saving."""
        return {"base_model_state": self.model.state_dict(), "bonus_model_state": self.bonus_model.state_dict()}

    def load_state_dict(self, state_dict):
        """Loads any base and/or bonus model parameters given."""
        if "base_model_state" in state_dict:
            self.model.load_state_dict(state_dict["base_model_state"])
        if "bonus_model_state" in state_dict:
            self.bonus_model.load_state_dict(state_dict["bonus_model_state"])

    def train_mode(self, itr):
        """Set both models in training mode (e.g. see PyTorch's ``Module.train()``)."""
        super().train_mode(itr)
        self.bonus_model.train()

    def sample_mode(self, itr):
        """Set both models in sampling mode."""
        super().sample_mode(itr)
        self.bonus_model.eval()

    def eval_mode(self, itr):
        """Set both models in evaluation mode."""
        super().eval_mode(itr)
        self.bonus_model.eval()

