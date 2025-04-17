import torch
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from random import randint

# from utils import reformat_img

from vit import ViT

import logging

logger = logging.getLogger(__name__)


class VitTrainer(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        # Save all hparams (e.g. the `cfg` object), so we can load checkpoints
        # without having to pass them to __init__ again. Do not print the hparams
        # to the logger to avoid clutter.
        self.save_hyperparameters(logger=False)

        self.cfg = cfg
        self.model = ViT(out_dim=cfg.num_puzzle_patches)

        self.crit = torch.nn.CrossEntropyLoss()
        self.register_buffer(
            'gt_single', torch.arange(cfg.num_puzzle_patches).unsqueeze(0)
        )

        self.cur_step_name_ = None

    def prefix_(self, name):
        return f'{self.cur_step_name_}/{name}'

    def log_(self, name, value, *args, **kwargs):
        self.log(self.prefix_(name), value, sync_dist=True, *args, **kwargs)

    def log_image_(self, key, images, **kwargs):
        # Only log images from the master process.
        if self.trainer.global_rank != 0:
            return
        if self.logger is not None:
            wb: WandbLogger = self.logger
            wb.log_image(self.prefix_(key), images, self.global_step, **kwargs)

    def forward(self, x):
        return self.model(x)

    def step_(self, batch, batch_idx):
        y = self(batch)  # y.shape = (B, num_patches, num_patches)
        gt = self.gt_single.expand(y.size(0), -1)
        loss = self.crit(y, gt)
        self.log_('loss', loss)
        return loss

    def on_train_epoch_start(self):
        self.cur_step_name_ = 'train'

    def on_validation_epoch_start(self):
        self.cur_step_name_ = 'val'
        # Pick a random batch from the val set to log images.
        num_val_batches = len(self.trainer.val_dataloaders)
        self.selected_batch_idx_ = randint(0, num_val_batches - 1)

    def training_step(self, *args):
        return self.step_(*args)

    def validation_step(self, *args):
        return self.step_(*args)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
