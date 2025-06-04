# -----------------------------------------------------------
# Inference script using DDIM sampling; saves figures to Test/
# -----------------------------------------------------------

import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import numpy as np
from contextlib import nullcontext
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from diffusers import UNet2DConditionModel, DDIMScheduler, DDPMScheduler  # ← DDIM!
from src.data.lrhr_dataset import LRHRDataSet  # make sure this module is import‑able

# -------------------------- CONFIG -------------------------- #
CKPT_PATH = "results/20250602_101735/checkpoints/best.pt"  # path to .pt checkpoint
BATCH_SIZE = 1  # dataloader batch size
NUM_SAMPLES = 30  # number of LR‒HR pairs to run
STEPS = 50  # DDIM inference steps
IMG_SIZE = 256  # UNet sample size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "results/20250602_101735/validation_samples"  # directory for .png outputs
ETA = 0.0  # DDIM eta; 0 ⇒ deterministic, >0 ⇒ stochastic
NUM_CLASSES = 11  # <-- Set this to match your training
# ------------------------------------------------------------ #

os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------- DATA ----------------------------- #
print("[INFO] Loading dataset …")

# Get total dataset and split into train/val using random_split with a fixed seed
# full_dataset = LRHRDataSet(mode="val")
# total_size = len(full_dataset)
# split = int(0.9 * total_size)
# lengths = [split, total_size - split]
# generator = torch.Generator().manual_seed(42)
# _, dataset = random_split(full_dataset, lengths, generator=generator)
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
full_dataset = LRHRDataSet(mode="train")
total_size = len(full_dataset)
print(f"[INFO] Total dataset size: {total_size}")
indices = np.arange(total_size)
np.random.seed(42)
np.random.shuffle(indices)
split = int(0.9 * total_size)
val_indices = indices[split:]
dataset = LRHRDataSet(mode="val", indices=val_indices)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Validation dataset size: {len(dataset)} samples")

# Compute min and max over the whole validation dataset (HR channel)
hr_min, hr_max = float("inf"), float("-inf")
for batch in dataloader:
    _, im_hr, _, _, _ = [t for t in batch]
    batch_min = im_hr.min().item()
    batch_max = im_hr.max().item()
    if batch_min < hr_min:
        hr_min = batch_min
    if batch_max > hr_max:
        hr_max = batch_max
print(f"[INFO] HR data range over validation set: min={hr_min}, max={hr_max}")

# ----------------- MODEL + SCHEDULER ------------------------ #
print("[INFO] Building model …")
# model = UNet2DConditionModel(
#     sample_size=IMG_SIZE,
#     in_channels=1,
#     out_channels=1,
#     layers_per_block=2,
#     block_out_channels=(32, 64, 128),
#     down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
#     up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
#     cross_attention_dim=3,
# ).to(DEVICE)

model = UNet2DConditionModel(
    sample_size=IMG_SIZE,
    in_channels=4 + NUM_CLASSES,  # HR noise + LR + one-hot + dem + ewc
    out_channels=1,
    layers_per_block=2,
    # block_out_channels=(64, 128, 256, 256),
    block_out_channels=(128, 256, 384, 512),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    class_embed_type=None,
    encoder_hid_dim=1,
).to(DEVICE)

scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_schedule="linear",  # matches DDPM default
    clip_sample=False,  # keep full dynamic range
)

# scheduler = DDPMScheduler(num_train_timesteps=1000)


# Move scheduler buffers to the same device as the model
try:  # diffusers ≥ 0.25 exposes .to()
    scheduler = scheduler.to(DEVICE)
except AttributeError:
    for k, v in scheduler.__dict__.items():
        if isinstance(v, torch.Tensor):
            setattr(scheduler, k, v.to(DEVICE))

print("[INFO] Loading checkpoint …")
checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model"], strict=True)
model.eval()

# scheduler = DDIMScheduler.from_config(checkpoint["scheduler"])
# scheduler.set_timesteps(STEPS, device=DEVICE)

# scheduler = DDPMScheduler.from_config(checkpoint["scheduler"])
# scheduler.set_timesteps(STEPS, device=DEVICE)

# -------------------- HELPERS ------------------------------- #


def lr_to_embedding(lr_up: torch.Tensor):
    b, c, h, w = lr_up.shape
    return lr_up.view(b, c, h * w).permute(0, 2, 1).to(DEVICE)


def denormalize(img, mean, std):
    return img * std + mean


def onehot_spatial(cond_lc, num_classes, device):
    # cond_lc: [B, 1, H, W] -> [B, NUM_CLASSES, H, W]
    cond_lc = (cond_lc / 10).to(torch.int64)
    cond_lc_onehot = torch.nn.functional.one_hot(
        cond_lc.squeeze(1), num_classes=num_classes
    )  # [B, H, W, C]
    cond_lc_onehot = (
        cond_lc_onehot.permute(0, 3, 1, 2).float().to(device)
    )  # [B, C, H, W]
    return cond_lc_onehot


# Automatic mixed precision only on CUDA
autocast_ctx = (
    torch.autocast(device_type="cuda") if DEVICE.type == "cuda" else nullcontext()
)

scheduler.set_timesteps(STEPS, device=DEVICE)

# -------------------- INFERENCE ----------------------------- #
print("[INFO] Starting inference with DDIM …")
with torch.no_grad():
    # Get means and stds from dataset for denormalization
    # means = dataset.dataset._means
    # stds = dataset.dataset._stds
    means = dataset._means
    stds = dataset._stds

    psnr_list = []
    ssim_list = []
    scc_list = []
    mse_list = []

    for batch_idx, batch in enumerate(dataloader):
        im_lr, im_hr, cond_angle, cond_lc, cond_dem = [
            t.to(DEVICE).float() for t in batch
        ]

        # One-hot encode cond_lc
        cond_lc_onehot = onehot_spatial(
            cond_lc, NUM_CLASSES, DEVICE
        )  # [B, NUM_CLASSES, H, W]
        cond_dem_normed = (cond_dem - 200) / 100

        # Start from pure Gaussian noise having HR shape
        x = torch.randn_like(im_hr).to(DEVICE)

        # Conditioning from LR
        cond_emb = lr_to_embedding(im_lr)

        # DDIM sampling loop, concatenate LR and one-hot at each step
        for t in scheduler.timesteps:
            t_batch = torch.full((x.shape[0],), t, device=DEVICE, dtype=torch.long)
            x_cond = torch.cat(
                [x, im_lr, cond_angle, cond_lc_onehot, cond_dem_normed], dim=1
            )  # [B, 2+NUM_CLASSES, H, W]
            with autocast_ctx:
                pred_noise = model(x_cond, t_batch, encoder_hidden_states=cond_emb)[
                    "sample"
                ]

            x = scheduler.step(
                model_output=pred_noise,
                timestep=t_batch,
                sample=x,
                eta=ETA,
            ).prev_sample

        # Clamp, move to CPU for plotting
        sr = x.cpu().squeeze(1)
        im_lr_cpu = im_lr.cpu().squeeze(1)
        im_hr_cpu = im_hr.cpu().squeeze(1)

        # --------- DENORMALIZE for visualization --------------
        # Channel 0: LR, Channel 1: HR
        sr_denorm = denormalize(sr, means[1], stds[1])
        im_lr_denorm = denormalize(im_lr_cpu, means[0], stds[0])
        im_hr_denorm = denormalize(im_hr_cpu, means[1], stds[1])

        # ------------- VISUALIZATION & SAVING --------------- #
        for i in range(sr.size(0)):
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(im_lr_denorm[i], cmap="viridis")
            axs[0].set_title("Low-Res (Input)")

            axs[1].imshow(im_hr_denorm[i], cmap="viridis")
            axs[1].set_title("High-Res (GT)")

            axs[2].imshow(sr_denorm[i], cmap="viridis")
            axs[2].set_title("Reconstructed HR (DDIM)")

            # ---- Convert to numpy for metrics ----
            hr_np = (
                im_hr_denorm[i].cpu().numpy()
                if torch.is_tensor(im_hr_denorm[i])
                else im_hr_denorm[i]
            )
            sr_np = (
                sr_denorm[i].cpu().numpy()
                if torch.is_tensor(sr_denorm[i])
                else sr_denorm[i]
            )

            # ---- Compute metrics ----
            psnr = compute_psnr(hr_np, sr_np, data_range=hr_max - hr_min)
            ssim = compute_ssim(hr_np, sr_np, data_range=hr_max - hr_min)
            scc = np.corrcoef(sr_np.flatten(), hr_np.flatten())[0, 1]
            mse = np.mean((sr_np - hr_np) ** 2)

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            scc_list.append(scc)
            mse_list.append(mse)

            # Centralized subtitle with tab-separated metrics
            fig.suptitle(
                f"PSNR: {psnr:.2f} dB   |   SSIM: {ssim:.3f}   |    SCC: {scc:.3f}   |   MSE: {mse:.4f}",
                fontsize=12,
                y=0.98,
            )

            for ax in axs:
                ax.axis("off")

            # Make layout more tight and images closer
            plt.subplots_adjust(
                wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.92, bottom=0.01
            )
            plt.tight_layout(rect=[0, 0, 1, 0.95], pad=0.2, w_pad=0.01, h_pad=0.05)

            fname = f"sample_{batch_idx}_{i}.png"
            fpath = os.path.join(SAVE_DIR, fname)
            fig.savefig(fpath, bbox_inches="tight")
            plt.close(fig)
            print(f"[INFO] Saved → {fpath}")

    # Print average metrics at the end
    print(f"\n[RESULT] Average PSNR: {np.mean(psnr_list):.2f} dB")
    print(f"[RESULT] Average SSIM: {np.mean(ssim_list):.4f}")
    print(f"[RESULT] Average SCC: {np.mean(scc_list):.4f}")
    print(f"[RESULT] Average MSE: {np.mean(mse_list):.4f}")

print("[DONE] All figures written to", os.path.abspath(SAVE_DIR))
