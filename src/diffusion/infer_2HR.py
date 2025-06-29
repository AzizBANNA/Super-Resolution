# -----------------------------------------------------------
# Inference script using DDIM sampling for dual-channel HR (VV + VH)
# -----------------------------------------------------------

import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from contextlib import nullcontext
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from diffusers import UNet2DConditionModel, DDIMScheduler
from src.data.lrhr_dataset_2HR import LRHRDataSet  



def denormalize(img, mean, std):
    return img * std + mean

# -------------------------- CONFIG -------------------------- #
DEVICE = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda")

CKPT_PATH = "results/20250624_230941/checkpoints/best.pt" 
BATCH_SIZE = 1
STEPS = 50
IMG_SIZE = 256
SAVE_DIR = "results/20250624_230941/validation_samples"
ETA = 0.0
NUM_CLASSES = 9
HR_CHANNELS = 2
# ------------------------------------------------------------ #

os.makedirs(SAVE_DIR, exist_ok=True)

print("[INFO] Loading dataset …")
full_dataset = LRHRDataSet(mode="train")
total_size = len(full_dataset)
indices = np.arange(total_size)
np.random.seed(42)
np.random.shuffle(indices)
split = int(0.9 * total_size)
val_indices = indices[split:]
dataset = LRHRDataSet(mode="val", indices=val_indices)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Validation dataset size: {len(dataset)} samples")

# Compute min/max over both VV and VH
print("[INFO] Computing global min/max of denormalized HR VV and VH channels …")
vv_min, vv_max = float("inf"), float("-inf")
vh_min, vh_max = float("inf"), float("-inf")

for batch in dataloader:
    _, hr_vv, hr_vh, _, _, _ = [t.to(DEVICE).float() for t in batch]

    hr_vv_denorm = denormalize(hr_vv, dataset._means[1], dataset._stds[1])  # VV stats
    hr_vh_denorm = denormalize(hr_vh, dataset._means[2], dataset._stds[2])  # VH stats

    vv_min = min(vv_min, hr_vv_denorm.min().item())
    vv_max = max(vv_max, hr_vv_denorm.max().item())
    vh_min = min(vh_min, hr_vh_denorm.min().item())
    vh_max = max(vh_max, hr_vh_denorm.max().item())

# print(f"[INFO] VV range: min={vv_min:.4f}, max={vv_max:.4f}")
# print(f"[INFO] VH range: min={vh_min:.4f}, max={vh_max:.4f}")

print("[INFO] Building model …")
model = UNet2DConditionModel(
    sample_size=IMG_SIZE,
    in_channels=HR_CHANNELS + 3 + NUM_CLASSES,
    out_channels=HR_CHANNELS,
    layers_per_block=2,
    block_out_channels=(128, 256, 384, 512),
    down_block_types=("DownBlock2D",)*4,
    up_block_types=("UpBlock2D",)*4,
    class_embed_type=None,
    encoder_hid_dim=1,
).to(DEVICE)

scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_schedule="linear",
    clip_sample=False,
)
# Set timesteps first
scheduler.set_timesteps(STEPS)

# Manually move all scheduler tensors to the correct device
for attr in dir(scheduler):
    val = getattr(scheduler, attr)
    if isinstance(val, torch.Tensor):
        setattr(scheduler, attr, val.to(DEVICE))

model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE)["model"])
model.eval()

# -------------------- HELPERS ------------------------------- #
def lr_to_embedding(lr_up):
    b, c, h, w = lr_up.shape
    return lr_up.view(b, c, h * w).permute(0, 2, 1).to(DEVICE)

def denormalize(img, mean, std):
    return img * std + mean

def onehot_spatial(cond_lc, num_classes, device):
    cond_lc = (cond_lc / 10).to(torch.int64)
    return torch.nn.functional.one_hot(cond_lc.squeeze(1), num_classes=num_classes).permute(0, 3, 1, 2).float().to(device)

autocast_ctx = torch.autocast(device_type="cuda") if DEVICE.type == "cuda" else nullcontext()

# -------------------- INFERENCE ----------------------------- #
print("[INFO] Starting inference with DDIM …")
with torch.no_grad():
    means = dataset._means
    stds = dataset._stds
    psnr_list, ssim_list = [], []

    for batch_idx, batch in enumerate(dataloader):
        im_lr, hr_vv, hr_vh, cond_angle, cond_lc, cond_dem = [t.to(DEVICE).float() for t in batch]
        im_hr = torch.cat([hr_vv, hr_vh], dim=1)
        cond_lc_onehot = onehot_spatial(cond_lc, NUM_CLASSES, DEVICE)
        cond_emb = lr_to_embedding(im_lr)
        cond_dem_normed = (cond_dem - 200) / 100

        x = torch.randn_like(im_hr).to(DEVICE)

        for t in scheduler.timesteps:
            t_batch = torch.full((x.shape[0],), t.item(), device=DEVICE, dtype=torch.long)
            x_cond = torch.cat([x, im_lr, cond_angle, cond_lc_onehot, cond_dem_normed], dim=1)
            #print(f"[DEBUG] scheduler.timesteps device: {scheduler.timesteps.device}")
            #print(f"[DEBUG] scheduler.alphas_cumprod device: {scheduler.alphas_cumprod.device}")
            with autocast_ctx:
                pred_noise = model(x_cond, t_batch, encoder_hidden_states=cond_emb)["sample"]
            x = scheduler.step(pred_noise, t_batch, x, eta=ETA).prev_sample

        # Clamp, move to CPU for plotting
        sr = x.cpu()
        im_lr_cpu = im_lr.cpu()
        im_hr_cpu = torch.cat([hr_vv, hr_vh], dim=1).cpu()

        # --------- DENORMALIZE for visualization --------------
        sr_denorm = denormalize(sr, means[1], stds[1])
        im_lr_denorm = denormalize(im_lr_cpu, means[0], stds[0])
        im_hr_denorm = denormalize(im_hr_cpu, means[1], stds[1])

        for i in range(sr.shape[0]):
            fig, axs = plt.subplots(1, HR_CHANNELS * 2 + 1, figsize=(18, 4))

            # Extract numpy arrays
            gt_vv = im_hr_denorm[i, 0].cpu().numpy()
            gt_vh = im_hr_denorm[i, 1].cpu().numpy()
            sr_vv = sr_denorm[i, 0].cpu().numpy()
            sr_vh = sr_denorm[i, 1].cpu().numpy()
            lr_img = im_lr_denorm[i, 0].cpu().numpy()

            # Compute metrics
            # For VV
            psnr_vv = compute_psnr(gt_vv, sr_vv, data_range=vv_max - vv_min)
            ssim_vv = compute_ssim(gt_vv, sr_vv, data_range=vv_max - vv_min)

            # For VH
            psnr_vh = compute_psnr(gt_vh, sr_vh, data_range=vh_max - vh_min)
            ssim_vh = compute_ssim(gt_vh, sr_vh, data_range=vh_max - vh_min)

            mse_vv = np.mean((gt_vv - sr_vv) ** 2)
            mse_vh = np.mean((gt_vh - sr_vh) ** 2)

            # Save for overall stats
            psnr_list.extend([psnr_vv, psnr_vh])
            ssim_list.extend([ssim_vv, ssim_vh])

            # Plotting
            axs[0].imshow(lr_img, cmap="viridis")
            axs[0].set_title("Input LR")

            axs[1].imshow(gt_vv, cmap="viridis")
            axs[1].set_title("GT VV")

            axs[2].imshow(gt_vh, cmap="viridis")
            axs[2].set_title("GT VH")

            axs[3].imshow(sr_vv, cmap="viridis")
            axs[3].set_title("SR VV")

            axs[4].imshow(sr_vh, cmap="viridis")
            axs[4].set_title("SR VH")

            for ax in axs:
                ax.axis("off")

            # Add super title with metrics
            fig.suptitle(
                f"VV → PSNR: {psnr_vv:.2f} dB | SSIM: {ssim_vv:.3f} | MSE: {mse_vv:.4f}    "
                f"VH → PSNR: {psnr_vh:.2f} dB | SSIM: {ssim_vh:.3f} | MSE: {mse_vh:.4f}",
                fontsize=12,
                y=0.98
            )

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            fname = f"sample_{batch_idx}_{i}.png"
            plt.savefig(os.path.join(SAVE_DIR, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"[INFO] Saved → {fname}")


    print(f"\n[RESULT] Avg PSNR: {np.mean(psnr_list):.2f} dB")
    print(f"[RESULT] Avg SSIM: {np.mean(ssim_list):.4f}")

print("[DONE] All figures written to", os.path.abspath(SAVE_DIR))
