import pytorch_lightning as pl
import torch


class BaseGenerativeModel(pl.LightningModule):
    """Base for one-step generative forecasting models."""

    fixed_cost = False  # True if ``sample`` ignores the schedule (the eval then runs a single budget)

    def _assign_init_args(self, init_locals: dict, *, ignore: tuple[str, ...] = ()) -> None:
        """Store __init__ arguments as attributes (skipping ``self`` and ``ignore``)."""
        for name, value in init_locals.items():
            # zero-arg super() puts a __class__ cell in locals(); assigning it
            # to self would downcast subclass instances
            if name in ("self", "__class__") or name in ignore:
                continue
            setattr(self, name, value)

    def _unpack(self, target: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Flatten the context and slice the target frame to the predicted channels.

        target: [B, 1, C, H, W] -> [B, C_out, H, W] (exogenous channels, e.g. det forcing, dropped)
        cond: [B, S, C, H, W] -> [B, S*C, H, W]
        """
        B, S, C, H, W = cond.shape
        return cond.view(B, S * C, H, W), target[:, 0, : self.net.out_channels]

    def sample(self, cond: torch.Tensor, *, sampling_schedule: tuple[str, int] = ("u", 1), **kwargs) -> torch.Tensor:
        """Sample one next state.

        cond: [B, C, H, W]
        returns: [B, C, H, W]
        """
        raise NotImplementedError

    def compute_loss(self, target: torch.Tensor, cond: torch.Tensor):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        cond, target = batch
        loss = self.compute_loss(target, cond)
        # No sync_dist: the epoch-end all-reduce clobbers the on_epoch accumulator
        # on multi-node (denormal/mixed-sign garbage). Without it, Lightning skips
        # the cross-rank reduce and logs rank-0's epoch mean, which is clean.
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        """No-op; metrics run in the eval callback."""
        return None

    test_step = validation_step

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
