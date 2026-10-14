"""Learning-rate schedule used by all training configs."""

import torch


class InverseLR(torch.optim.lr_scheduler.LRScheduler):
    """lr(step) = (1 - warmup^(step+1)) * max(final_lr, base_lr * (1 + step/inv_gamma)^-power).

    `warmup` in [0, 1) sets an exponential warm-up (0 disables it); the decay halves
    the learning rate (for power=1) after `inv_gamma` steps.
    """

    def __init__(self, optimizer, inv_gamma=1.0, power=1.0, warmup=0.0, final_lr=0.0,
                 last_epoch=-1):
        if not 0.0 <= warmup < 1:
            raise ValueError("warmup must be in [0, 1)")
        self.inv_gamma, self.power, self.warmup, self.final_lr = inv_gamma, power, warmup, final_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        ramp = 1 - self.warmup ** (step + 1)
        decay = (1 + step / self.inv_gamma) ** -self.power
        return [ramp * max(self.final_lr, lr * decay) for lr in self.base_lrs]
