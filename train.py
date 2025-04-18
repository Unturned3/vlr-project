import hydra
from omegaconf import OmegaConf

import torch
import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from utils import LockStepWandbLogger

from vit_trainer import VitTrainer

from dataset import ImageNetDataset

import os
from pathlib import Path
import logging
import warnings


# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

logger = logging.getLogger(__name__)


@rank_zero_only
def log_hparams(wb_logger, cfg):
    if not cfg.fast_dev_run:
        # Log hyperparameters to W&B
        cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wb_logger.log_hyperparams(cfg_dict)

        # Save the config locally too
        project_name = wb_logger.name
        run_id = wb_logger.experiment.id
        run_log_dir = Path(cfg.log.save_dir) / project_name / run_id
        os.makedirs(run_log_dir, exist_ok=True)
        OmegaConf.save(cfg, run_log_dir / 'config.yaml')


@hydra.main(config_path='config', config_name='train', version_base='1.3')
def main(cfg):
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

    wb_logger = LockStepWandbLogger(project=cfg.project_name, save_dir=cfg.log.save_dir)
    log_hparams(wb_logger, cfg)

    early_stop = hydra.utils.instantiate(cfg.early_stop)
    save_best_ckpt = hydra.utils.instantiate(cfg.save_best_ckpt)
    logger.info(f'ckpt dir: {save_best_ckpt.dirpath}')

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=-1,  # Use all available GPUs (DDP is used automatically)
        fast_dev_run=cfg.fast_dev_run,
        max_epochs=cfg.num_epochs,
        enable_progress_bar=True,
        log_every_n_steps=cfg.log.freq,
        logger=wb_logger,
        callbacks=[early_stop, save_best_ckpt],
    )
    wb_logger.set_trainer_(trainer)

    trainer.fit(
        model=vit_trainer,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=cfg.resume_from_ckpt,
    )


if __name__ == '__main__':
    main()
