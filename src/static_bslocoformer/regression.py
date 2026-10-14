import torch
from torch import nn


class Regressor(nn.Module):
    """
    Regression-based model wrapper
    """

    def __init__(
        self,
        backbone: nn.Module,
        loss: nn.Module,
    ):
        super().__init__()

        self.backbone = backbone
        self.loss_fn = loss

    def forward(self, x):
        """
        Args:
            x: torch.Tensor
                Mixture signal (B, C, T)

        Returns:
            s: torch.Tensor
                Separated source stems (B, N_src, C, T)
        """

        return self.backbone(x)

    def train_step(self, s, q = None, debug: bool = False, *args, **kwargs):
        """
        Args:
            s: torch.Tensor
                Source stems (B, N_src, C, T)

            q: For compatibility with query trainer.

        Returns:
            loss_dict: dict
                Losses in a dictionary {loss, loss_linear, loss_unnorm}
        """

        mix = s.sum(dim=1)

        pred = self.forward(mix)
        target = s

        loss_unnorm = (pred - target).pow(2).mean(dim=(1, 2))
        loss_norm = loss_unnorm / (target.pow(2).mean(dim=(1, 2)) + 1e-8)

        loss_linear = loss_norm.mean()
        loss_unnorm = loss_unnorm.mean()

        loss = self.loss_fn(pred, target)

        return {
            "loss": loss,
            "loss_linear": loss_linear,
            "loss_unnorm": loss_unnorm,
        }

    @torch.no_grad()
    def sample(
        self,
        mix: torch.Tensor | None = None,
        sources: torch.Tensor | None = None,
        debug: bool = False,
        *args,
        **kwargs,
    ):
        """
        mix: (B, C, T)
        sources: (B, N_src, C, T)
        """
        assert mix is None or sources is None, "Provide exactly one of mix or sources."

        if sources is not None:
            mix = sources.sum(dim=1)  # (B, C, T)

        assert mix is not None, "Provide either mix or sources."

        x = self.forward(mix)

        if debug and sources is not None:
            info = {}
            info["sources"] = sources
            return x, info

        return x
