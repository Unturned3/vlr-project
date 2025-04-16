import hydra
from omegaconf import OmegaConf

import torch
import lightning as pl
from utils import LockStepWandbLogger

from vit_trainer import VitTrainer

from dataset import ImageNetDataset

import os
import logging

logger = logging.getLogger(__name__)


@hydra.main(config_path='config', config_name='train', version_base='1.3')
def main(cfg):
    logger.info('Preparing...')
    dataset = ImageNetDataset(cfg.data_dir, cfg.image_paths_pkl, cfg.image_size)

    # The entire ImageNet dataset (1.2M images) is too large. We
    # will use a (fixed) random subset for initial experiments.
    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(cfg.rand_subset_seed)
    )
    subset = torch.utils.data.Subset(dataset, indices[: cfg.rand_subset_size])

    ds_size = len(subset)
    train_size = int(cfg.train_percent * ds_size)
    val_size = ds_size - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        subset, [train_size, val_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    vit_trainer = VitTrainer(cfg)

    os.makedirs(cfg.log.save_dir, exist_ok=True)
    wb_logger = LockStepWandbLogger(
        project=cfg.wandb.project, save_dir=cfg.log.save_dir
    )

    if not cfg.fast_dev_run:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wb_logger.log_hyperparams(cfg_dict)

    trainer = pl.Trainer(
        accelerator='gpu',
        # devices=-1,  # Use all available GPUs
        # strategy='ddp',
        fast_dev_run=cfg.fast_dev_run,
        max_epochs=cfg.num_epochs,
        enable_progress_bar=True,
        log_every_n_steps=cfg.log.freq,
        logger=wb_logger,
    )
    wb_logger.set_trainer_(trainer)

    logger.info('Starting training...')
    trainer.fit(
        model=vit_trainer,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == '__main__':
    main()
