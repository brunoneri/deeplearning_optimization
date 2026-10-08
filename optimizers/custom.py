import torch

from optimizers._base import _Base


class AdaBelief(_Base):
    """AdaBelief optimizer that adapts to the prediction error variance."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16):
        """Initialize AdaBelief moment estimates and epsilon."""
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    def _update(self, group, p, g, state):
        """Apply the AdaBelief update using the belief-based second moment."""
        lr, (b1, b2), eps = group["lr"], group["betas"], group["eps"]
        state["t"] = t = state.get("t", 0) + 1
        m = self._buf(state, "m", p)
        s = self._buf(state, "s", p)
        m.lerp_(g, 1 - b1)
        diff = g - m
        s.mul_(b2).addcmul_(diff, diff, value=1 - b2).add_(eps)
        m_hat = m / (1 - b1 ** t)
        s_hat = s / (1 - b2 ** t)
        p.addcdiv_(m_hat, s_hat.sqrt() + eps, value=-lr)


class Lion(_Base):
    """Lion optimizer using sign updates and decoupled weight decay."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        """Initialize the Lion learning rate and momentum-like parameters."""
        super().__init__(params, dict(lr=lr, betas=betas,
                                      weight_decay=weight_decay))

    def _update(self, group, p, g, state):
        """Apply the Lion sign-gradient update."""
        lr, (b1, b2) = group["lr"], group["betas"]
        m = self._buf(state, "m", p)
        c = torch.lerp(m, g, 1 - b1)
        p.mul_(1 - lr * group["weight_decay"])
        p.add_(c.sign(), alpha=-lr)
        m.lerp_(g, 1 - b2)