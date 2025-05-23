# -----------------------------------------------------------
# Inference script using DDIM sampling; saves figures to Test/
# -----------------------------------------------------------

import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from contextlib import nullcontext

from diffusers import UNet2DConditionModel, DDIMScheduler, DDPMScheduler  # ← DDIM!
from src.data.lrhr_dataset import LRHRDataSet  # make sure this module is import‑able

# -------------------------- CONFIG -------------------------- #
CKPT_PATH = "results/diffusion_baseline_checkpoints/diff_sr_ep499.pt"  # path to .pt checkpoint
BATCH_SIZE = 1  # dataloader batch size
NUM_SAMPLES = 30  # number of LR‒HR pairs to run
STEPS = 50  # DDIM inference steps
IMG_SIZE = 256  # UNet sample size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "results/diffusion_baseline_samples"  # directory for .png outputs
ETA = 0.0  # DDIM eta; 0 ⇒ deterministic, >0 ⇒ stochastic
# ------------------------------------------------------------ #

os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------- DATA ----------------------------- #
print("[INFO] Loading dataset …")
dataset = LRHRDataSet(mode="train", indices=np.arange(30))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

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
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 256),
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


# Automatic mixed precision only on CUDA
autocast_ctx = (
    torch.autocast(device_type="cuda") if DEVICE.type == "cuda" else nullcontext()
)

scheduler.set_timesteps(STEPS, device=DEVICE)

# -------------------- INFERENCE ----------------------------- #
print("[INFO] Starting inference with DDIM …")
with torch.no_grad():
    for batch_idx, batch in enumerate(dataloader):
        # ---------------------------------------------------- #
        im_lr, im_hr, im_cond1, im_cond2 = [t.to("cuda").float() for t in batch[:4]]

        # im_lr = im_lr.to(DEVICE, dtype=torch.float32)
        # im_hr = im_hr.to(DEVICE, dtype=torch.float32)

        # network_input =  torch.cat([im_lr_norm, im_cond1, im_cond2_normed], dim=1)
        # Start from pure Gaussian noise having HR shape
        x = torch.randn_like(im_hr).to(DEVICE)
        # x = im_cond1

        # Conditioning from LR
        cond_emb = lr_to_embedding(im_lr)

        # Iterate over scheduler‑generated timesteps
        for t in scheduler.timesteps:
            t_batch = torch.full((x.shape[0],), t, device=DEVICE, dtype=torch.long)
            with autocast_ctx:
                pred_noise = model(x, t_batch, encoder_hidden_states=cond_emb)["sample"]
            # DDIM update (eta controls stochasticity)
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

        # ------------- VISUALIZATION & SAVING --------------- #
        for i in range(sr.size(0)):
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(im_lr_cpu[i], cmap="viridis")
            axs[0].set_title("Low‑Res (Input)")

            axs[1].imshow(
                im_hr_cpu[i], cmap="viridis"
            )  # Or denormalize this too if needed
            axs[1].set_title("High‑Res (GT)")

            axs[2].imshow(sr[i], cmap="viridis")
            axs[2].set_title("Reconstructed HR (DDIM)")

            for ax in axs:
                ax.axis("off")
            plt.tight_layout()

            fname = f"sample_{batch_idx}_{i}.png"
            fpath = os.path.join(SAVE_DIR, fname)
            fig.savefig(fpath, bbox_inches="tight")
            plt.close(fig)
            print(f"[INFO] Saved → {fpath}")

print("[DONE] All figures written to", os.path.abspath(SAVE_DIR))
