import torch
from dataset import ImageDataset
from vit_trainer import VitTrainer

import matplotlib.pyplot as plt

from utils import (
    permute_image,
    invert_permutation,
    hungarian_refine,
    visualize_errors,
)

from dataclasses import dataclass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

ckpt_path = 'logs/duj9f3yt-729-e60.ckpt'
vit_trainer = VitTrainer.load_from_checkpoint(ckpt_path).to(device)

cfg = vit_trainer.cfg
vit = vit_trainer.model.eval()
dataset = ImageDataset(cfg.data_dir, cfg.image_paths_pkl)


@dataclass
class Result:
    orig_image: torch.Tensor
    perm_image: torch.Tensor
    perm: torch.Tensor
    raw_recon: torch.Tensor
    raw_err: torch.Tensor
    refined_recon: torch.Tensor
    refined_err: torch.Tensor


def load_puzzle(idx):
    orig_image = vit_trainer.resize_image(dataset[idx])
    perm_image, perm = permute_image(orig_image.clone(), patch_size=cfg.patch_size)
    return orig_image, perm_image, perm


def run(orig_image, perm_image, perm):
    with torch.inference_mode():
        x = perm_image.unsqueeze(0).to(device)
        x = vit_trainer.norm_image(x)
        pred_perm_logits = vit(x)
        pred_perm = pred_perm_logits.squeeze().argmax(dim=-1)

    gt = vit_trainer.gt_single[:, perm].expand(pred_perm_logits.size(0), -1)
    gt_perm = gt.squeeze()

    # Raw reconstruction
    inv_perm = invert_permutation(pred_perm, allow_degenerate=True)
    raw_recon, _ = permute_image(
        perm_image.clone(), cfg.patch_size, inv_perm, allow_degenerate=True
    )

    raw_err = visualize_errors(raw_recon, cfg.patch_size, pred_perm, gt_perm, alpha=0.4)

    # Refined reconstruction
    refined_pred_perm = hungarian_refine(pred_perm_logits.squeeze())

    refined_inv_perm = invert_permutation(refined_pred_perm)
    refined_recon, _ = permute_image(perm_image, cfg.patch_size, refined_inv_perm)
    refined_err = visualize_errors(
        refined_recon, cfg.patch_size, refined_pred_perm, gt_perm
    )

    return Result(
        orig_image,
        perm_image,
        perm,
        raw_recon,
        raw_err,
        refined_recon,
        refined_err,
    )


# idx = 50398  # easy car
# idx = 90186  # almost uniform texture
# idx = 153060  # large white sky
# idx = 39549  # overexposed van
# idx = 8583  # weird visualization error: white patches aren't labeled as wrong. (fixed?)
# idx = 79616  # easy bird
# idx = 13553  # pond with white sky
# idx = 27528 # easy city skyline
# idx = 446944 # black and white image
# idx = 129884 # white bg wall


# Generate 3 good examples

indices = [50398, 90186, 79616]

rs: list[Result] = []
for idx in indices:
    orig_image, perm_image, perm = load_puzzle(idx)
    r = run(orig_image, perm_image, perm)
    rs.append(r)

fig, axs = plt.subplots(3, 2, figsize=(4, 6))

for ax in axs.ravel():
    ax.axis('off')

for i, r in enumerate(rs):
    axs[i, 0].imshow(r.perm_image.permute(1, 2, 0))
    axs[i, 1].imshow(r.refined_recon.permute(1, 2, 0))

fig.tight_layout()
fig.savefig('plots/good_examples.png', dpi=200)
plt.close(fig)


# 2 examples showing interesting errors after refinement

indices = [13553, 153060]

rs: list[Result] = []
for idx in indices:
    orig_image, perm_image, perm = load_puzzle(idx)
    r = run(orig_image, perm_image, perm)
    rs.append(r)

fig, axs = plt.subplots(2, 2, figsize=(4, 4))

for ax in axs.ravel():
    ax.axis('off')

for i, r in enumerate(rs):
    axs[i, 0].imshow(r.refined_recon.permute(1, 2, 0))
    axs[i, 1].imshow(r.refined_err)

fig.tight_layout()
fig.savefig('plots/degenerate_errors.png', dpi=200)
plt.close(fig)


# 2 examples showing what the decoder does

indices = [39549, 129884]

rs: list[Result] = []
for idx in indices:
    orig_image, perm_image, perm = load_puzzle(idx)
    r = run(orig_image, perm_image, perm)
    rs.append(r)

fig, axs = plt.subplots(2, 2, figsize=(4, 4))

for ax in axs.ravel():
    ax.axis('off')

for i, r in enumerate(rs):
    # axs[i, 0].imshow(r.raw_recon.permute(1, 2, 0))
    axs[i, 0].imshow(r.raw_err)
    axs[i, 1].imshow(r.refined_recon.permute(1, 2, 0))

fig.tight_layout()
fig.savefig('plots/decoder_effect.png', dpi=200)
plt.close(fig)
