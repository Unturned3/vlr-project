import torch
import accelerate
import accelerate.logging
from accelerate.utils import InitProcessGroupKwargs
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
import psutil
import atexit
import time


class Trainer:
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
        self.model = self.aclr.prepare(model)

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
        self.gt_single = gt_single.to(self.aclr.device)

        # Optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.learning_rate,
        )
        self.optimizer = self.aclr.prepare(optimizer)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=cfg.learning_rate,
            total_steps=cfg.num_epochs * len(self.train_loader),
            pct_start=cfg.lr_pct_start,
            anneal_strategy=cfg.lr_anneal_strategy,
            cycle_momentum=False,  # No momentum in Adam
        )

    def init_infrastructure(self):
        cfg = self.cfg
        self.global_step = 0
        self.cur_step_name_ = None

        logging.basicConfig(
            level=logging.INFO,
            format=r'%(asctime)s %(levelname)s: %(message)s',
            datefmt=r'%y-%m-%d %H:%M:%S',
        )

        ipg_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=15))

        self.aclr = accelerate.Accelerator(kwargs_handlers=[ipg_kwargs])
        self.logger = accelerate.logging.get_logger(__name__)
        self.caught_signal = torch.tensor(0, device=self.aclr.device, dtype=torch.int)

        if self.aclr.is_main_process:
            slurm_job_id = os.environ.get('SLURM_JOB_ID', 'local')
            master_pid_file = f'.master.pid.{slurm_job_id}'

            def sig_handler(signum, _):
                self.caught_signal.fill_(signum)

            def cleanup():
                if os.path.exists(master_pid_file):
                    os.remove(master_pid_file)

            atexit.register(cleanup)

            with open(master_pid_file, 'w') as f:
                f.write(f'{os.getpid()}\n')

            for sig in [
                signal.SIGINT,
                signal.SIGUSR1,
            ]:
                signal.signal(sig, sig_handler)

        self.logger.info(f'SLURM_JOB_ID: {os.environ.get("SLURM_JOB_ID", "None")}')
        self.logger.info(f'PID: {os.getpid()}', main_process_only=False)

        # Populate runtime-computed fields in the config
        cfg.num_patches = (cfg.crop_image_to_px // cfg.patch_size) ** 2

        if self.aclr.is_main_process:
            self.wb_run = wandb.init(
                project=cfg.project_name,
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                dir=cfg.log.save_dir,
                resume='allow',  # TODO: implement resuming from given run_id
            )
            self.logger.info(f'W&B run id: {self.wb_run.id}')
            timestamp = datetime.datetime.now().strftime(r'%y%m%d-%H%M%S')
            self.log_dir = Path(cfg.log.save_dir) / f'{self.wb_run.id}-{timestamp}'
            os.makedirs(self.log_dir)
            OmegaConf.save(cfg, self.log_dir / 'config.yaml')

            # Remove all files in ~/msg
            os.system('rm -rf ~/msg/*')

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

        if cfg.batch_size % self.aclr.num_processes != 0:
            self.logger.warning(
                f'Batch size {cfg.batch_size} is not divisible by the world size '
                f'{self.aclr.num_processes}. This may lead to uneven batch '
                f'sizes across processes.'
            )

        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=cfg.batch_size // self.aclr.num_processes,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=lambda x: x,  # Do nothing. We perform the transforms & batching all on the GPU.
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=cfg.batch_size // self.aclr.num_processes,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=lambda x: x,  # Do nothing. See above.
        )
        self.train_loader = self.aclr.prepare(train_loader)
        self.val_loader = self.aclr.prepare(val_loader)

    def interruptible_iter(self, iterator):
        """Wraps an iterator to allow for graceful shutdowns."""
        for item in iterator:
            if (self.global_step + 1) % self.cfg.check_interrupt_period == 0:
                accelerate.utils.broadcast(self.caught_signal, from_process=0)
                if self.caught_signal.item() != 0:
                    self.logger.info(
                        'Early exit, stopping training.', main_process_only=False
                    )
                    raise EarlyExit
            yield item

    def log(self, name, value, period=1, sync_dist=True):
        # TODO: reimplement this function to "lazily" store values to be logged
        # When we actually want the log to be pushed to the W&B server, we should
        # call something like `log_flush()`, which "batches" all tensors that needs
        # to be reduced across processes, then calls `wandb.log()`.
        if (self.global_step + 1) % period != 0:
            return
        # if not self.accelerator.is_local_main_process:
        #    return
        if isinstance(value, torch.Tensor):
            value = value.detach()
            if sync_dist:
                value = self.aclr.reduce(value, reduction='mean')
            value = value.item()
        if self.aclr.is_main_process:
            self.wb_run.log({name: value}, step=self.global_step)

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
            for batch_idx, batch in enumerate(
                self.interruptible_iter(self.train_loader)
            ):
                self.optimizer.zero_grad()
                loss, t1_acc, t5_acc = self._step(batch, batch_idx)
                self.log('train/loss', loss, period=cfg.log.freq)
                self.log('train/t1_acc', t1_acc, period=cfg.log.freq)
                self.log('train/t5_acc', t5_acc, period=cfg.log.freq)
                self.log(
                    'lr-Adam',
                    self.scheduler.get_last_lr()[0],
                    period=cfg.log.freq,
                )
                self.aclr.backward(loss)
                self.optimizer.step()
                self.scheduler.step()
                self.global_step += 1

            self.model.eval()
            with torch.no_grad():
                loss_sum, t1_acc_sum, t5_acc_sum = 0, 0, 0
                for batch_idx, batch in enumerate(
                    self.interruptible_iter(self.val_loader)
                ):
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
        wb_exit_code = 0
        try:
            self._main_loop()
        except EarlyExit:
            proc = psutil.Process(os.getpid())
            if self.caught_signal.item() == signal.SIGINT:
                wb_exit_code = 255  # Will show as "Killed" on W&B
                if self.aclr.is_main_process:
                    with open(f'/home/yusenh/msg/{proc.pid}.txt', 'a') as f:
                        f.write(f'Process {proc.pid} ({proc.name()}) caught SIGINT.\n')
            elif self.caught_signal.item() == signal.SIGUSR1:
                wb_exit_code = 1  # Will show as "Preempted" on W&B
                if self.aclr.is_main_process:
                    self.wb_run.mark_preempting()
                    with open(f'/home/yusenh/msg/{proc.pid}.txt', 'a') as f:
                        f.write(f'Process {proc.pid} ({proc.name()}) caught SIGUSR1.\n')
        except Exception as e:
            wb_exit_code = 2  # Will show as "Failed" on W&B
            proc = psutil.Process(os.getpid())
            with open(f'/home/yusenh/msg/{proc.pid}.txt', 'a') as f:
                f.write(f'Process {proc.pid} ({proc.name()}) caught Exception: {e}\n')

        proc = psutil.Process(os.getpid())
        with open(f'/home/yusenh/msg/{proc.pid}.txt', 'a') as f:
            f.write(f'Process {proc.pid} ({proc.name()}) waiting for everyone.\n')

        if self.aclr.is_main_process:
            self.wb_run.finish(wb_exit_code)

        self.aclr.wait_for_everyone()
        self.aclr.end_training()

        with open(f'/home/yusenh/msg/{proc.pid}.txt', 'a') as f:
            f.write(f'Process {proc.pid} ({proc.name()}) finished.\n')


class EarlyExit(Exception):
    pass


@hydra.main(config_path='config', config_name='train', version_base='1.3')
def main(cfg):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == '__main__':
    main()
