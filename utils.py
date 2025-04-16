from einops import rearrange

import torch

from lightning.pytorch.loggers import WandbLogger
from typing_extensions import override
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.fabric.utilities.logger import _add_prefix

# import imageio


def move_to_device(dct, device):
    for key, value in dct.items():
        if isinstance(value, torch.Tensor):
            dct[key] = value.to(device)
    return dct


# def write_mp4(images, filename, chan_first=False, flip=False):
#    writer = imageio.get_writer(filename, fps=30)
#    for image in images:
#        if chan_first:
#            image = rearrange(image, 'c h w -> h w c')
#        if flip:
#            image = image[::-1]
#        writer.append_data(image)
#    writer.close()


def reformat_img(x):
    x = rearrange(x, 'c h w -> h w c')
    # Without clipping, extreme values can cause the image to be displayed as all black.
    x = x.clip(0, 1)
    return x.cpu().detach().numpy()


class LockStepWandbLogger(WandbLogger):
    """
    Wrapper around WandbLogger that syncs the wandb step with Lightning global_step.

    Currently, the wandb web UI only supports "steps" on the slider bar of logged
    images. If steps aren't synced with global_step, the slider bar will be difficult
    to interpret.
    """

    def set_trainer_(self, trainer):
        self.trainer_ = trainer

    @override
    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'

        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)

        self.experiment.log(
            dict(metrics, **{'trainer/global_step': self.trainer_.global_step}),
            step=self.trainer_.global_step,
        )
