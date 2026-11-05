# -----------------------------------------------------------
# Inference for the single-channel (VV) model using DDIM sampling.
# Saves one figure per validation sample plus per-region metrics.
# -----------------------------------------------------------

import argparse
import os
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader
import numpy as np
from contextlib import nullcontext
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from collections import defaultdict

from diffusers import UNet2DConditionModel, DDIMScheduler
from src.data.lrhr_dataset import LRHRDataSet

# -------------------------- CONFIG -------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
parser.add_argument("--data_root", default="sr_data", help="Dataset folder")
parser.add_argument("--save_dir", default="results/validation_samples", help="Output folder")
args = parser.parse_args()

CKPT_PATH = args.ckpt
DATA_ROOT = args.data_root
SAVE_DIR = args.save_dir
BATCH_SIZE = 1  # dataloader batch size
STEPS = 50  # DDIM inference steps
IMG_SIZE = 256  # UNet sample size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ETA = 0.0  # DDIM eta; 0 ⇒ deterministic, >0 ⇒ stochastic
NUM_CLASSES = 9  # must match training

# ----------------------------SET SEEDS and dirs ------------- #
torch.manual_seed(42)
np.random.seed(42)
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------- DATA ----------------------------- #
print("[INFO] Loading dataset …")
full_dataset = LRHRDataSet(mode="train", data_root=DATA_ROOT, zscore=False)
total_size = len(full_dataset)
print(f"[INFO] Total dataset size: {total_size}")

indices = np.arange(total_size)
np.random.shuffle(indices)
split = int(0.9 * total_size)

train_indices = indices[:split]
dataset_train = LRHRDataSet(
    mode="train", data_root=DATA_ROOT, indices=train_indices, augment=False
)

val_indices = indices[split:]
dataset = LRHRDataSet(
    mode="val",
    data_root=DATA_ROOT,
    indices=val_indices,
    augment=False,
    means=dataset_train._means,
    stds=dataset_train._stds,
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Validation dataset size: {len(dataset)} samples")

def denormalize(img, mean, std):
    return img * std + mean


# Compute min and max over the whole validation dataset (HR channel)
hr_global_min, hr_global_max = float("inf"), float("-inf")
for batch in dataloader:
    _, im_hr, _, _, _, _ = [t for t in batch]
    im_hr_denorm = denormalize(
        im_hr, dataset_train._means[1], dataset_train._stds[1]
    )  # denormalize HR
    batch_min = im_hr_denorm.min().item()
    batch_max = im_hr_denorm.max().item()
    if batch_min < hr_global_min:
        hr_global_min = batch_min
    if batch_max > hr_global_max:
        hr_global_max = batch_max
print(
    f"[INFO] HR data range over validation set: min={hr_global_min}, max={hr_global_max}"
)

# ----------------- MODEL + SCHEDULER ------------------------ #
print("[INFO] Building model …")
model = UNet2DConditionModel(
    sample_size=IMG_SIZE,
    in_channels=3 + NUM_CLASSES,  # noisy HR + LR + DEM + one-hot land cover
    out_channels=1,
    layers_per_block=2,
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

# -------------------- HELPERS ------------------------------- #


def lr_to_embedding(lr_up: torch.Tensor):
    b, c, h, w = lr_up.shape
    return lr_up.view(b, c, h * w).permute(0, 2, 1).to(DEVICE)


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
    # Training set statistics, used for denormalization
    means = dataset_train._means
    stds = dataset_train._stds

    psnr_list = []
    ssim_list = []
    scc_list = []
    mse_list = []

    # Initialize dictionaries to store metrics per region
    region_metrics = defaultdict(lambda: {"psnr": [], "ssim": [], "scc": [], "mse": []})

    for batch_idx, batch in enumerate(dataloader):
        im_lr, im_hr, _, cond_lc, cond_dem, region_timestamp = [
            t.to(DEVICE).float() if isinstance(t, torch.Tensor) else t for t in batch
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
                [x, im_lr, cond_lc_onehot, cond_dem_normed], dim=1
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

        # Compute the range for sr_denorm
        sr_min = min(sr.min() for sr in sr_denorm)
        sr_max = max(sr.max() for sr in sr_denorm)

        # Compute the range for hr_denorm
        hr_min = min(hr.min() for hr in im_hr_denorm)
        hr_max = max(hr.max() for hr in im_hr_denorm)

        # Take the min of both minimum values and the max of both maximum values
        global_min = min(sr_min, hr_min)
        global_max = max(sr_max, hr_max)

        # ------------- VISUALIZATION & SAVING --------------- #
        for i in range(sr.size(0)):
            fig = plt.figure(figsize=(16, 4))  # Increase width to accommodate histogram

            # Adjust GridSpec to include space for the histogram
            gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 1])  # Remove separate colorbar column

            axs = [fig.add_subplot(gs[j]) for j in range(3)]
            hist_ax = fig.add_subplot(gs[3])  # Dedicated axis for the histogram

            lr = axs[0].imshow(im_lr_denorm[i], cmap="viridis")
            axs[0].set_title("LR")

            hr = axs[1].imshow(
                im_hr_denorm[i], cmap="viridis", vmin=global_min, vmax=global_max
            )
            axs[1].set_title("GT HR")

            rec = axs[2].imshow(
                sr_denorm[i], cmap="viridis", vmin=global_min, vmax=global_max
            )
            axs[2].set_title("Rec. HR")

            # Add the colorbar directly to the reconstructed image plot
            plt.colorbar(rec, ax=axs[2], fraction=0.046, pad=0.04)  # Adjust size and padding
            plt.colorbar(hr, ax=axs[1], fraction=0.046, pad=0.04)  # Adjust size and padding
            plt.colorbar(lr, ax=axs[0], fraction=0.046, pad=0.04)  # Adjust size and padding

            # ---- Add histogram to the right ----
            hist_ax.hist(im_hr_denorm[i].cpu().numpy().flatten(), bins=100, range=(global_min.cpu().numpy(), global_max.cpu().numpy()), alpha=0.5, label="GT HR")
            hist_ax.hist(sr_denorm[i].cpu().numpy().flatten(), bins=100, range=(global_min.cpu().numpy(), global_max.cpu().numpy()), alpha=0.5, label="Rec. HR")
            hist_ax.set_title("Pixel Value Histogram")
            hist_ax.legend(loc="upper right", fontsize=8)  # Make legend smaller

            # Remove y-axis from the histogram
            hist_ax.yaxis.set_visible(False)
            hist_ax.tick_params(axis="x", labelsize=8)  # Make x-axis ticks smaller

            # Match the aspect ratio of the histogram to the HR image
            hist_ax.set_box_aspect(1)  # Ensure the aspect ratio matches the HR image

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
            psnr = compute_psnr(hr_np, sr_np, data_range=hr_global_max - hr_global_min)
            ssim = compute_ssim(hr_np, sr_np, data_range=hr_global_max - hr_global_min)
            scc = np.corrcoef(sr_np.flatten(), hr_np.flatten())[0, 1]
            mse = np.mean((sr_np - hr_np) ** 2)

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            scc_list.append(scc)
            mse_list.append(mse)

            # Split region_timestamp into region and timestamp, and remove .tif extension
            region, timestamp = region_timestamp[i].split("/")
            timestamp = timestamp.replace(".tif", "")

            # Store metrics per region
            region_metrics[region]["psnr"].append(psnr)
            region_metrics[region]["ssim"].append(ssim)
            region_metrics[region]["scc"].append(scc)
            region_metrics[region]["mse"].append(mse)

            # Centralized subtitle with region and timestamp on the first line, and metrics on the second line
            fig.suptitle(
                f"Region: {region}   |   Date: {timestamp}\n"
                f"PSNR: {psnr:.2f} dB   |   SSIM: {ssim:.3f}   |   SCC: {scc:.3f}   |   MSE: {mse:.4f}",
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

            fpath = os.path.join(SAVE_DIR, f"{region}_{timestamp}.png")
            fig.savefig(fpath, bbox_inches="tight")
            plt.close(fig)
            print(f"[INFO] Saved → {fpath}")

    # Print average metrics at the end
    print(f"\n[RESULT] Average PSNR: {np.mean(psnr_list):.2f} dB")
    print(f"[RESULT] Average SSIM: {np.mean(ssim_list):.4f}")
    print(f"[RESULT] Average SCC: {np.mean(scc_list):.4f}")
    print(f"[RESULT] Average MSE: {np.mean(mse_list):.4f}")

    # Compute average metrics per region
    print("\n[RESULT] Metrics per region:")
    for region, metrics in region_metrics.items():
        avg_psnr = np.mean(metrics["psnr"])
        avg_ssim = np.mean(metrics["ssim"])
        avg_scc = np.mean(metrics["scc"])
        avg_mse = np.mean(metrics["mse"])
        print(f"Region: {region}")
        print(f"  Average PSNR: {avg_psnr:.2f} dB")
        print(f"  Average SSIM: {avg_ssim:.4f}")
        print(f"  Average SCC: {avg_scc:.4f}")
        print(f"  Average MSE: {avg_mse:.4f}")

    # Convert defaultdict to regular dict before saving
    region_metrics_dict = dict(region_metrics)

    # Save region_metrics dictionary as a .npy file
    metrics_save_path = os.path.join(SAVE_DIR, "region_metrics.npy")
    np.save(metrics_save_path, region_metrics_dict)
    print(f"[INFO] Saved region metrics dictionary to {metrics_save_path}")

print("[DONE] All figures written to", os.path.abspath(SAVE_DIR))
