import torch
import accelerate
import accelerate.logging
import numpy as np
import random
from random import randint

# from utils import reformat_img
from utils import topk_acc

from vit import ViT

import hydra
from omegaconf import OmegaConf, DictConfig
import wandb
import logging

import torchvision.transforms.v2 as vT

from dataset import ImageDataset

import os
from pathlib import Path
import datetime

import signal


class Trainer:
    caught_signals: set[signal.Signals] = set()

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.torch_rng_seed)
        np.random.seed(cfg.numpy_rng_seed)
        random.seed(cfg.python_rng_seed)

        self.init_infrastructure()  # Loggers, W&B, config saving, signal handlers, etc.

        self.init_dataloaders()

        model = ViT(
            patch_size=cfg.patch_size,
            dim=cfg.patch_emb_dim,
            out_dim=cfg.num_patches,
            depth=cfg.num_layers,
            heads=cfg.num_heads,
            dim_head=cfg.head_dim,
        )
        self.model = self.accelerator.prepare(model)

        # Image transforms
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

        # Loss functions
        self.criterion = torch.nn.CrossEntropyLoss()

        gt_single = torch.arange(self.cfg.num_patches).unsqueeze(0)
        self.gt_single = gt_single.to(self.accelerator.device)

        # Optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.learning_rate,
        )
        self.optimizer = self.accelerator.prepare(optimizer)

    def init_infrastructure(self):
        cfg = self.cfg
        self.global_step = 0
        self.cur_step_name_ = None

        def sig_handler(signum, _frame):
            Trainer.caught_signals.add(signum)

        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGUSR1, signal.SIGUSR2]:
            signal.signal(sig, sig_handler)

        logging.basicConfig(
            level=logging.INFO,
            format=r'%(asctime)s %(levelname)s: %(message)s',
            datefmt=r'%y-%m-%d %H:%M:%S',
        )

        self.accelerator = accelerate.Accelerator(log_with='wandb')
        self.logger = accelerate.logging.get_logger(__name__)

        self.logger.info(f'SLURM_JOB_ID: {os.environ.get("SLURM_JOB_ID", "None")}')
        self.logger.info(f'PID: {os.getpid()}', main_process_only=False)

        # Populate runtime-computed fields in the config
        cfg.num_patches = (cfg.crop_image_to_px // cfg.patch_size) ** 2

        # Initialize Weights & Biases
        self.accelerator.init_trackers(
            project_name=cfg.project_name,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            init_kwargs={
                'wandb': {
                    'dir': cfg.log.save_dir,
                    'resume': 'allow',  # TODO: implement resuming from given run_id
                }
            },
        )

        if self.accelerator.is_main_process:
            wandb_run_id = self.accelerator.get_tracker('wandb').run.id
            timestamp = datetime.datetime.now().strftime(r'%y%m%d-%H%M%S')
            self.logger.info(f'W&B run id: {wandb_run_id}')
            # FIXME: how do other ranks access log_dir?
            # Maybe this isn't a problem, if only rank 0 saves checkpoints, etc.
            self.log_dir = Path(cfg.log.save_dir) / f'{wandb_run_id}-{timestamp}'
            os.makedirs(self.log_dir)
            OmegaConf.save(cfg, self.log_dir / 'config.yaml')

    def init_dataloaders(self):
        cfg = self.cfg

        self.dataset = ImageDataset(
            root_dir=cfg.data_dir,
            image_paths_pkl=cfg.image_paths_pkl,
        )

        if cfg.rand_subset_size != -1:
            indices = torch.randperm(
                len(self.dataset),
                generator=torch.Generator().manual_seed(cfg.rand_subset_seed),
            )
            subset = torch.utils.data.Subset(
                self.dataset, indices[: cfg.rand_subset_size]
            )
        else:
            subset = self.dataset

        train_size = int(cfg.train_percent * len(subset))
        val_size = len(subset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            subset, [train_size, val_size]
        )

        # FIXME: divide batch_size by world_size if using DDP
        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=lambda x: x,  # Do nothing. We perform the transforms & batching all on the GPU.
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=lambda x: x,  # Do nothing. See above.
        )
        self.train_loader = self.accelerator.prepare(train_loader)
        self.val_loader = self.accelerator.prepare(val_loader)

    def log(self, name, value, period=1, sync_dist=True):
        if (self.global_step + 1) % period != 0:
            return
        if isinstance(value, torch.Tensor):
            value = value.detach()
            if sync_dist:
                value = self.accelerator.reduce(value, reduction='mean')
            value = value.item()
        self.accelerator.log({name: value}, step=self.global_step)

    # TODO: Adapt this implementation
    # def log_image_(self, key, images, **kwargs):  # noqa: ARG002
    #    if not self.accelerator.is_main_process:
    #        return
    #    wandb_tracker = self.accelerator.get_tracker('wandb')
    #    if wandb_tracker:
    #        wandb_tracker.log(
    #            {self.prefix_(key): [wandb.Image(img) for img in images]},
    #            step=self.global_step,
    #        )

    def _step(self, batch, _batch_idx):
        # Tensors should already be on self.accelerator.device
        batch = [self.resize_and_norm(img) for img in batch]
        batch = torch.stack(batch)
        y = self.model(batch)  # y.shape = (B, num_patches, num_patches)
        gt = self.gt_single.expand(y.size(0), -1)
        loss = self.criterion(y, gt)
        t1_acc = topk_acc(y, gt, k=1)
        t5_acc = topk_acc(y, gt, k=5)
        return loss, t1_acc, t5_acc

    def _main_loop(self):
        cfg = self.cfg
        for epoch in range(cfg.num_epochs):
            self.model.train()
            for batch_idx, batch in enumerate(interruptible(self.train_loader)):
                self.optimizer.zero_grad()
                loss, t1_acc, t5_acc = self._step(batch, batch_idx)
                self.log('train/loss', loss, period=cfg.log.freq)
                self.log('train/t1_acc', t1_acc, period=cfg.log.freq)
                self.log('train/t5_acc', t5_acc, period=cfg.log.freq)
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.global_step += 1

            self.model.eval()
            with torch.no_grad():
                loss_sum, t1_acc_sum, t5_acc_sum = 0, 0, 0
                for batch_idx, batch in enumerate(interruptible(self.val_loader)):
                    loss, t1_acc, t5_acc = self._step(batch, batch_idx)
                    loss_sum += loss.detach()
                    t1_acc_sum += t1_acc.detach()
                    t5_acc_sum += t5_acc.detach()

            self.log('val/loss', loss_sum / len(self.val_loader), period=1)
            self.log('val/t1_acc', t1_acc_sum / len(self.val_loader), period=1)
            self.log('val/t5_acc', t5_acc_sum / len(self.val_loader), period=1)

            self.log('epoch', epoch)
            self.logger.info(f'Epoch {epoch + 1}/{cfg.num_epochs} finished.')

    def run(self):
        try:
            self._main_loop()
        except GracefulShutdown:
            # TODO: implement checkpointing, etc.
            self.logger.warning('Graceful shutdown requested. Exiting...')
        # Mark training as finished ONLY if
        # 1. self._main_loop() finished without exceptions
        # 2. GracefulShutdown was raised.
        # If any other exceptions occur, it will be shown as "Failed" on W&B.
        self.accelerator.end_training()


class GracefulShutdown(Exception):
    pass


def interruptible(iterator):
    """Wraps any iterator to allow for graceful shutdowns."""
    for item in iterator:
        if Trainer.caught_signals:
            raise GracefulShutdown
        yield item


@hydra.main(config_path='config', config_name='train', version_base='1.3')
def main(cfg):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == '__main__':
    main()
