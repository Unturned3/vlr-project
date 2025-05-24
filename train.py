import torch
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from random import randint

# from utils import reformat_img
from utils import topk_acc

from vit import ViT

import hydra
import logging

import torchvision.transforms.v2 as vT

from utils import permute_image

logger = logging.getLogger(__name__)


class VitTrainer(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        # Save all hparams (e.g. the `cfg` object), so we can load checkpoints
        # without having to pass them to __init__ again. Do not print the hparams
        # to the logger to avoid clutter.
        self.save_hyperparameters(logger=False)

        self.cfg = cfg

        assert cfg.crop_image_to_px % cfg.patch_size == 0

        self.num_patches = (cfg.crop_image_to_px // cfg.patch_size) ** 2

        self.model = ViT(
            patch_size=cfg.patch_size,
            dim=cfg.patch_emb_dim,
            out_dim=self.num_patches,
            depth=cfg.num_layers,
            heads=cfg.num_heads,
            dim_head=cfg.head_dim,
        )

        self.resize_image = vT.Compose(
            [
                vT.Resize(cfg.resize_image_to_px, antialias=True),
                vT.CenterCrop(cfg.crop_image_to_px),
            ]
        )

        self.norm_image = vT.Compose(
            [
                vT.ToDtype(torch.float32, scale=True),  # uint8 [0, 255] -> f32 [0, 1]
                vT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.resize_and_norm = vT.Compose([self.resize_image, self.norm_image])

        self.crit = torch.nn.CrossEntropyLoss()
        self.register_buffer('gt_single', torch.arange(self.num_patches).unsqueeze(0))

        self.cur_step_name_ = None

    # Return the metric name prefixed with the current step name
    def prefix_(self, name):
        return f'{self.cur_step_name_}/{name}'

    # Log metrics prefixed with the current step name
    def log_(self, name, value, *args, **kwargs):
        # sync_dist=True: average metrics across all GPUs before logging
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

    def step_(self, batch, _batch_idx):
        # Tensors should already be on self.device
        batch = [self.resize_and_norm(img) for img in batch]
        batch = torch.stack(batch)

        y = self(batch)  # y.shape = (B, num_patches, num_patches)
        gt = self.gt_single.expand(y.size(0), -1)
        loss = self.crit(y, gt)
        self.log_('t1_acc', topk_acc(y, gt, k=1))
        self.log_('t5_acc', topk_acc(y, gt, k=5))
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
        scheduler = hydra.utils.instantiate(
            self.cfg.lr_scheduler.scheduler,
            optimizer=optimizer,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self.cfg.lr_scheduler.monitor,
                'interval': 'epoch',
                'frequency': 1,
            },
        }
