
import torch


class SelfSupervisedModule(torch.nn.Module):
    """
    Designates derived modules as being self-supervised.
    Intended to allow other components (e.g. agent) to assume the
    forward call will also produce the module's own loss.

    Since Python is dynamically typed, this cannot be strictly
    enforced. It is up to the developer to maintain this pattern.
    """

    def forward(self, *input) -> (torch.Tensor, torch.Tensor) or ((torch.Tensor,), torch.Tensor):
        """
        Should have return format: (output, loss)
        where output is an arbitrary Tensor or tuple of Tensors,
        and loss is a scalar Tensor.
        """
        raise NotImplementedError
