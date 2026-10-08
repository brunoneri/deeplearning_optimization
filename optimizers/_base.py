import torch
from torch.optim import Optimizer


class _Base(Optimizer):
    """Common optimizer logic shared by all custom optimizers in this project."""

    @torch.no_grad()
    def step(self, closure=None):
        """Apply one optimizer step to all parameter groups.

        If a closure is provided, it is evaluated under grad mode before the
        update is applied to each parameter.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    self._update(group, p, p.grad, self.state[p])
        return loss

    @staticmethod
    def _buf(state, name, like):
        """Return a persistent state tensor, creating it when it does not exist."""
        if name not in state:
            state[name] = torch.zeros_like(like)
        return state[name]