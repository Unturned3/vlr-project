from einops import rearrange

import torch

from lightning.pytorch.loggers import WandbLogger
from typing_extensions import override
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.fabric.utilities.logger import _add_prefix

# import imageio

import torch
from einops import rearrange, repeat
import numpy as np

from scipy.optimize import linear_sum_assignment


def invert_permutation(perm: torch.Tensor, allow_degenerate: bool = False):
    """
    perm: (N,)
        A permutation of the integers 0, 1, ..., N - 1.
        Returns inv_perm such that perm[inv_perm] == [0, 1, ..., N - 1].

        allow_degenerate: bool
            If a target location has never been predicted,
            it'll be -1. This can be used to retrieve a last "placeholder"
            patch from an array of size N + 1, e.g. an all-black patch.
    """
    if allow_degenerate:
        inv_perm = torch.full_like(perm, -1, dtype=int)
    else:
        inv_perm = torch.zeros_like(perm)
    for i in range(len(perm)):
        inv_perm[perm[i]] = i
    return inv_perm


def greedy_refine_pred_perm(pred_perm_logits: torch.Tensor):
    """
    pred_perm: (n_patches, n_positions)
        In the future we might test the model by giving it less than
        n_patches (which it was trained on).
    """
    logits = pred_perm_logits.clone()
    pred_perm = logits.softmax(dim=-1)
    n_patches = pred_perm.shape[0]
    n_positions = pred_perm.shape[1]
    refined_perm = torch.full((n_patches,), -1, dtype=int)
    pos_occupied = torch.full((n_positions,), False, dtype=bool)
    finalized_cnt = 0
    while finalized_cnt < n_patches:
        confidence, pred_patch_idx = pred_perm.max(dim=-1)
        i = confidence.argmax()
        p_i = pred_patch_idx[i]
        # Input patch i is predicted to be in position p_i.
        if pos_occupied[p_i]:
            # p_i occupied by a previous higher-confidence prediction
            pred_perm[i, p_i] = 0
            continue
        else:
            # Position prediction for patch i is finalized.
            refined_perm[i] = p_i
            pred_perm[i, :] = 0  # Prevent subsequent selection of patch i.
            pos_occupied[p_i] = True
            finalized_cnt += 1
    return refined_perm


def refine_2(pred_perm: torch.Tensor):
    p = pred_perm.clone().softmax(dim=-1)
    n_patches, n_positions = p.shape
    refined = torch.full((n_patches,), -1, dtype=torch.long, device=p.device)
    row_used = torch.zeros(n_patches, dtype=torch.bool, device=p.device)
    col_used = torch.zeros(n_positions, dtype=torch.bool, device=p.device)

    for _ in range(min(n_patches, n_positions)):
        # mask out used rows and cols
        p[row_used, :] = -float('inf')
        p[:, col_used] = -float('inf')

        # global argmax
        flat_idx = p.view(-1).argmax()
        if p.view(-1)[flat_idx] == -float('inf'):
            break

        i = flat_idx // n_positions
        j = flat_idx % n_positions

        refined[i] = j
        row_used[i] = True
        col_used[j] = True

    return refined


def hungarian_refine(pred_perm: torch.Tensor):
    # minimize cost, so we pass -pred_perm
    cost = -pred_perm.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    # row_ind is [0,1,2,…], so col_ind[r] is the assigned column for row r
    refined = np.zeros_like(col_ind)
    refined[row_ind] = col_ind
    return torch.tensor(refined, dtype=torch.long, device=pred_perm.device)


def visualize_errors(
    recon_image, patch_size, pred_perm: torch.tensor, gt_perm: torch.tensor, alpha=0.4
):
    recon_image = recon_image.squeeze()
    recon_image = rearrange(recon_image, 'c h w -> h w c')
    n_side = recon_image.shape[0] // patch_size
    errors = torch.where(pred_perm != gt_perm)[0]

    r = errors // n_side
    c = errors % n_side

    overlay = torch.zeros((n_side, n_side, 3), dtype=torch.float32)
    overlay[:] = torch.tensor([0, 1, 0], dtype=torch.float32)
    overlay[r, c] = torch.tensor([1, 0, 0], dtype=torch.float32)

    inv_gt_perm = invert_permutation(gt_perm)
    overlay = rearrange(overlay, 'h w c -> (h w) c')
    overlay = overlay[inv_gt_perm, :]
    overlay = rearrange(overlay, '(h w) c -> h w c', h=n_side)

    overlay = repeat(overlay, 'h w c -> (h p1) (w p2) c', p1=patch_size, p2=patch_size)
    alpha = 0.4
    overlay = torch.clamp(alpha * overlay + (1 - alpha) * recon_image / 255, 0, 1)

    return overlay


def permute_image(
    img: torch.Tensor,
    patch_size: int,
    perm: torch.Tensor | None = None,
    allow_degenerate: bool = False,
    generator: torch.Generator | None = None,
):
    """
    Split a *square* image into non-overlapping patches, randomly permute the
    patches, and stitch them back together.

    Parameters
    ----------
    img : torch.Tensor (..., C, H, W)
        The input image.  H == W must hold.
    patch_size : int
        Side length of a square patch. Must exactly divide H and W.
    perm: torch.Tensor (n_patches,)
        The permutation to apply. If None, a random permutation is generated.
    generator : torch.Generator, optional
        For reproducible shuffling (`torch.Generator().manual_seed(…)`).

    Returns
    -------
    permuted_img : torch.Tensor (..., C, H, W)
    perm          : torch.Tensor (n_patches,)
    """
    *_, H, W = img.shape
    if H != W:
        raise ValueError('Image must be square (H == W)')
    if H % patch_size != 0:
        raise ValueError('patch_size must evenly divide the image size')

    n = H // patch_size  # patches per side, total patches = n²

    # (... C, H, W) -> (..., n², C, ph, pw)
    patches = rearrange(
        img,
        '... c (h ph) (w pw) -> ... (h w) c ph pw',
        h=n,
        w=n,
        ph=patch_size,
        pw=patch_size,
    )

    if allow_degenerate:
        black_patch = torch.zeros_like(patches[..., 0:1, :, :, :])
        patches = torch.cat([patches, black_patch], dim=-4)

    if perm is None:
        perm = torch.randperm(n * n, generator=generator, device=img.device)
    patches = patches[..., perm, :, :, :]

    # (..., n², C, ph, pw) -> (..., C, H, W)
    permuted_img = rearrange(
        patches,
        '... (h w) c ph pw -> ... c (h ph) (w pw)',
        h=n,
        w=n,
        ph=patch_size,
        pw=patch_size,
    )

    return permuted_img, perm


def unnormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Convert a normalized image tensor to a NumPy array for visualization with matplotlib.

    Args:
        tensor (torch.Tensor): A (3, H, W) tensor normalized with ImageNet stats.
        mean (list): List of mean values used for normalization.
        std (list): List of std dev values used for normalization.

    Returns:
        np.ndarray: A (H, W, 3) array in range [0, 1], suitable for plt.imshow.
    """
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError('Expected tensor shape (3, H, W)')

    unnormalized = tensor.clone().cpu()
    for c in range(3):
        unnormalized[c] = unnormalized[c] * std[c] + mean[c]

    unnormalized = torch.clamp(unnormalized, 0.0, 1.0)
    return unnormalized.permute(1, 2, 0).numpy()


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
