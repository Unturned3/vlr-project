from einops import rearrange

import torch

from lightning.pytorch.loggers import WandbLogger
from typing_extensions import override
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.fabric.utilities.logger import _add_prefix

# import imageio

import torch
from einops import rearrange


def permute_image(
    img: torch.Tensor,
    patch_size: int,
    perm: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
):
    """
    Split a *square* image into non-overlapping patches, randomly permute the
    patches, and stitch them back together.

    Parameters
    ----------
    img : torch.Tensor (C, H, W)
        The input image.  H == W must hold.
    patch_size : int
        Side length of a square patch. Must exactly divide H and W.
    perm: torch.Tensor (n_patches,)
        The permutation to apply. If None, a random permutation is generated.
    generator : torch.Generator, optional
        For reproducible shuffling (`torch.Generator().manual_seed(…)`).

    Returns
    -------
    permuted_img : torch.Tensor (C, H, W)
    perm          : torch.Tensor (n_patches,)
    """
    if img.ndim != 3:
        raise ValueError('img must have shape (C, H, W)')
    C, H, W = img.shape
    if H != W:
        raise ValueError('Image must be square (H == W)')
    if H % patch_size != 0:
        raise ValueError('patch_size must evenly divide the image size')

    n = H // patch_size  # patches per side, total patches = n²

    # (C, H, W) -> (n², C, ph, pw)
    patches = rearrange(
        img,
        'c (h ph) (w pw) -> (h w) c ph pw',
        h=n,
        w=n,
        ph=patch_size,
        pw=patch_size,
    )

    if perm is None:
        perm = torch.randperm(n * n, generator=generator, device=img.device)
    patches = patches[perm]

    # (n², C, ph, pw) -> (C, H, W)
    permuted_img = rearrange(
        patches,
        '(h w) c ph pw -> c (h ph) (w pw)',
        h=n,
        w=n,
        ph=patch_size,
        pw=patch_size,
    )

    return permuted_img, perm


def topk_acc(y, gt, k=1):
    """
    y: (..., batch, n_class)
    gt: (..., batch)
    """
    topk = y.topk(k, dim=-1).indices
    correct = (topk == gt.unsqueeze(-1)).any(dim=-1)
    accuracy = correct.float().mean()
    return accuracy


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
