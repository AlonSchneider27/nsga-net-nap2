"""Learning rate warmup scheduler.

Linear warmup from initial_lr to target_lr over warmup_epochs,
then delegates to an after_scheduler (e.g. StepLR).

Matches the WarmUpLR from the original NAP2 ModelTrainer.
"""
from __future__ import annotations

from torch.optim.lr_scheduler import LRScheduler


class WarmUpLR(LRScheduler):
    """Linear warmup then hand off to another scheduler."""

    def __init__(self, optimizer, after_scheduler, warmup_epochs=5,
                 initial_lr=1e-6, target_lr=1e-3):
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.after_scheduler = after_scheduler
        self.warmup_finished = False
        super().__init__(optimizer)

        # Set initial learning rates
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr

    def get_lr(self):
        if not self.warmup_finished:
            lr = [self.initial_lr + (self.target_lr - self.initial_lr)
                  * self.last_epoch / self.warmup_epochs
                  for _ in self.base_lrs]
            if self.last_epoch >= self.warmup_epochs:
                self.warmup_finished = True
                if self.after_scheduler:
                    self.after_scheduler.base_lrs = [
                        self.target_lr for _ in self.base_lrs
                    ]
            return lr
        if self.after_scheduler:
            return self.after_scheduler.get_last_lr()
        return [self.target_lr for _ in self.base_lrs]

    def step(self, epoch=None):
        if not self.warmup_finished:
            super().step()
        if self.warmup_finished and self.after_scheduler:
            self.after_scheduler.step()
