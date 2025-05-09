import hydra
from omegaconf import OmegaConf

import numpy as np
import random

import torch
import lightning as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_info
from utils import LockStepWandbLogger

from vit_trainer import VitTrainer

from dataset import ImageDataset

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
        cfg.num_patches = (cfg.crop_image_to_px // cfg.patch_size) ** 2

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
    torch.manual_seed(cfg.torch_rng_seed)
    np.random.seed(cfg.numpy_rng_seed)
    random.seed(cfg.python_rng_seed)

    wb_logger = LockStepWandbLogger(project=cfg.project_name, save_dir=cfg.log.save_dir)
    # Trigger the creation of the actual run, so any subsequent prints
    # to stdout are also logged on W&B.
    rank_zero_info(f'Run id: {wb_logger.experiment.id}')

    dataset = ImageDataset(
        root_dir=cfg.data_dir,
        image_paths_pkl=cfg.image_paths_pkl,
    )

    if cfg.rand_subset_size != -1:
        indices = torch.randperm(len(dataset))
        subset = torch.utils.data.Subset(dataset, indices[: cfg.rand_subset_size])
    else:
        subset = dataset

    ds_size = len(subset)
    train_size = int(cfg.train_percent * ds_size)
    val_size = ds_size - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        subset, [train_size, val_size]
    )

    # Do nothing. We perform the transforms & batching all on the GPU.
    def collate(batch):
        return batch

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate,
    )

    vit_trainer = VitTrainer(cfg)

    # Resume from checkpoint if specified
    # Filter out incompatible items (e.g. different classifier head dimensions)
    if cfg.resume_from_ckpt is not None:
        # weights_only=False, because the checkpoint has other stuff like OmegaConf dict
        # ckpt = torch.load(cfg.resume_from_ckpt, map_location='cpu', weights_only=False)
        # old_sd = ckpt['state_dict']
        old_sd = VitTrainer.load_from_checkpoint(
            cfg.resume_from_ckpt,
            map_location='cpu',
        ).state_dict()
        new_sd = vit_trainer.state_dict()
        filtered = {
            k: v
            for k, v in old_sd.items()
            if k in new_sd and v.shape == new_sd[k].shape
        }
        msg = vit_trainer.load_state_dict(filtered, strict=False)
        rank_zero_info(f'\nLoaded checkpoint: {cfg.resume_from_ckpt}')
        rank_zero_info(f'Ignored keys: {msg}\n')

    log_hparams(wb_logger, cfg)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stop = hydra.utils.instantiate(cfg.early_stop)
    save_best_ckpt = hydra.utils.instantiate(cfg.save_best_ckpt)

    rank_zero_info(f'SLURM_JOB_ID: {os.environ.get("SLURM_JOB_ID", "N/A")}')
    rank_zero_info(
        f'SLURM_ARRAY_TASK_ID: {os.environ.get("SLURM_ARRAY_TASK_ID", "N/A")}'
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=-1,  # Use all available GPUs (DDP is used automatically)
        fast_dev_run=cfg.fast_dev_run,
        max_epochs=cfg.num_epochs,
        enable_progress_bar=True,
        log_every_n_steps=cfg.log.freq,
        logger=wb_logger,
        callbacks=[
            early_stop,
            save_best_ckpt,
            lr_monitor,
        ],
    )
    wb_logger.set_trainer_(trainer)

    trainer.fit(
        model=vit_trainer,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == '__main__':
    main()
