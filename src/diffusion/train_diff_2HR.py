# Diffusion-based RGB super-resolution trainer with DDIM sampling (Dual-channel VV+VH version)

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tensorboardX import SummaryWriter
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from accelerate import Accelerator
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from src.data.lrhr_dataset_2HR import LRHRDataSet
from torchvision.utils import save_image
import os
import datetime

np.random.seed(42)  # For reproducibility

# ------------------------- hyperparameters ------------------------- #
NUM_EPOCHS = 500
SAVE_FREQ = 50
BATCH_SIZE = 6
LR = 5e-5
LOG_PATH = "results/diffusion_train_logs"
CKPT_DIR = "results/diffusion_baseline_checkpoints"
IMG_SIZE_HR = 256
USE_MIXED_PREC = True

NUM_CLASSES = 9  # 
HR_CHANNELS = 2  # <-- VV + VH


# ------------------------- helpers ------------------------- #
def build_accelerator():
    try:
        return Accelerator(mixed_precision="fp16" if USE_MIXED_PREC else "no")
    except TypeError:
        return Accelerator(fp16=USE_MIXED_PREC)


def lr_to_embedding(lr_up: torch.Tensor):
    b, c, h, w = lr_up.shape
    return lr_up.view(b, c, h * w).permute(0, 2, 1)


def sanity_check(im_lr, im_hr):
    if im_hr.shape[1] != HR_CHANNELS or im_lr.shape[1] != 1:
        raise ValueError(f"Expected HR with {HR_CHANNELS} channels and LR with 1")
    if im_hr.shape[-2:] != (256, 256):
        raise ValueError(f"HR image size is not 256x256 but {im_hr.shape[-2:]}")
    if torch.isnan(im_lr).any() or torch.isnan(im_hr).any():
        raise ValueError("Input contains NaN values")


def onehot_spatial(cond_lc, num_classes, device):
    cond_lc = (cond_lc / 10).to(torch.int64)
    cond_lc_onehot = F.one_hot(cond_lc.squeeze(1), num_classes=num_classes)
    cond_lc_onehot = cond_lc_onehot.permute(0, 3, 1, 2).float().to(device)
    return cond_lc_onehot


# ------------------------- trainer ------------------------- #
class DiffusionSRTrainer:
    def __init__(self, resume_ckpt=None):
        self.acc = build_accelerator()
        self.device = self.acc.device

        # ========== results directory with timestamp ==========
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join("results", timestamp)
        self.ckpt_dir = os.path.join(self.results_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.outputs_dir = os.path.join(self.results_dir, "outputs_baseline")
        os.makedirs(self.outputs_dir, exist_ok=True)
        self.log_path = os.path.join(self.results_dir, "diffusion_train_logs")

        # ========== dataset ==========
        full_dataset = LRHRDataSet(mode="train")
        total_size = len(full_dataset)
        print(f"[INFO] Total dataset size: {total_size}")
        indices = np.arange(total_size)
        np.random.seed(42)
        np.random.shuffle(indices)
        split = int(0.9 * total_size)
        train_indices, val_indices = indices[:split], indices[split:]

        train_dataset = LRHRDataSet(mode="train", indices=train_indices)
        val_dataset = LRHRDataSet(mode="val", indices=val_indices)

        self.train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
        )

        # ========== model ==========
        self.model = UNet2DConditionModel(
            sample_size=IMG_SIZE_HR,
            in_channels=HR_CHANNELS + 3 + NUM_CLASSES,
            out_channels=HR_CHANNELS,
            layers_per_block=2,
            block_out_channels=(128, 256, 384, 512),
            down_block_types=("DownBlock2D",)*4,
            up_block_types=("UpBlock2D",)*4,
            class_embed_type=None,
            encoder_hid_dim=1,
        )

        self.scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.opt = optim.AdamW(self.model.parameters(), lr=LR)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=NUM_EPOCHS)
        self.writer = SummaryWriter(self.log_path)
        self.step = self.epoch = 0
        self.best_val_loss = float("inf")

        self.model, self.opt, self.train_loader, self.val_loader = self.acc.prepare(
            self.model, self.opt, self.train_loader, self.val_loader
        )

        # ========== Resume checkpoint ==========
        if resume_ckpt is not None and os.path.isfile(resume_ckpt):
            print(f"[INFO] Loading checkpoint from {resume_ckpt}")
            checkpoint = torch.load(resume_ckpt, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.opt.load_state_dict(checkpoint["opt"])
            self.epoch = checkpoint.get("epoch", 0)
            self.step = checkpoint.get("step", 0)
            print(f"[INFO] Resumed from epoch {self.epoch}, step {self.step}")
            for _ in range(self.epoch):
                self.lr_scheduler.step()

    def train_step(self, batch):
        im_lr, im_hr_vv, im_hr_vh, cond_angle, cond_lc, cond_dem = batch
        im_hr = torch.cat([im_hr_vv, im_hr_vh], dim=1)  # Stack to shape (B, 2, 256, 256)
        im_lr = im_lr.to(self.device).float()
        im_hr = im_hr.to(self.device).float()
        cond_lc = cond_lc.to(self.device)
        cond_dem = cond_dem.to(self.device)

        sanity_check(im_lr, im_hr)

        b = im_hr.shape[0]
        t = torch.randint(0, self.scheduler.config.num_train_timesteps, (b,), device=self.device).long()
        noise = torch.randn_like(im_hr)
        x_t = self.scheduler.add_noise(im_hr, noise, t)

        cond_lc_onehot = onehot_spatial(cond_lc, NUM_CLASSES, self.device)
        cond_dem_normed = (cond_dem - 200) / 100
        x_t_cond = torch.cat([x_t, im_lr, cond_angle, cond_lc_onehot, cond_dem_normed], dim=1)
        cond_emb = lr_to_embedding(im_lr)

        with self.acc.autocast():
            pred_noise = self.model(x_t_cond, t, encoder_hidden_states=cond_emb)["sample"]
            loss = F.mse_loss(pred_noise, noise)

        self.acc.backward(loss)
        self.opt.step()
        self.opt.zero_grad()
        return loss.detach()

    def run(self):
        for self.epoch in range(self.epoch, NUM_EPOCHS):
            self.model.train()
            pbar = tqdm(self.train_loader)
            for batch in pbar:
                loss = self.train_step(batch)
                pbar.set_description(f"Epoch {self.epoch} | Loss: {loss.item():.4f}")
                if self.step % 50 == 0:
                    self.writer.add_scalar("train/loss", loss.item(), self.step)
                self.step += 1
            val_loss = self.validate()
            tqdm.write(f"Epoch {self.epoch} | Validation Loss: {val_loss:.4f}")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_ckpt(best=True)
            if (self.epoch + 1) % SAVE_FREQ == 0:
                self.save_ckpt()
            self.lr_scheduler.step()
        s#elf.sample_ddim(split="val")

    def save_ckpt(self, best=False):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        fpath = os.path.join(self.ckpt_dir, f"diff_sr_ep{self.epoch}.pt")
        torch.save({
            "model": self.acc.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "epoch": self.epoch,
            "step": self.step,
        }, fpath)
        if best:
            best_path = os.path.join(self.ckpt_dir, "best.pt")
            torch.save({
                "model": self.acc.get_state_dict(self.model),
                "opt": self.opt.state_dict(),
                "epoch": self.epoch,
                "step": self.step,
            }, best_path)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        count = 0
        for batch in self.val_loader:
            im_lr, im_hr_vv, im_hr_vh, cond_angle, cond_lc, cond_dem = [t.to(self.device) for t in batch]
            im_hr = torch.cat([im_hr_vv, im_hr_vh], dim=1) 
            sanity_check(im_lr, im_hr)
            cond_lc_onehot = onehot_spatial(cond_lc, NUM_CLASSES, self.device)
            cond_dem_normed = (cond_dem - 200) / 100

            b = im_hr.shape[0]
            t = torch.randint(0, self.scheduler.config.num_train_timesteps, (b,), device=self.device).long()
            noise = torch.randn_like(im_hr)
            x_t = self.scheduler.add_noise(im_hr, noise, t)
            x_t_cond = torch.cat([x_t, im_lr, cond_angle, cond_lc_onehot, cond_dem_normed], dim=1)
            cond_emb = lr_to_embedding(im_lr)
            pred = self.model(x_t_cond, t, encoder_hidden_states=cond_emb)["sample"]
            loss = F.mse_loss(pred, noise, reduction="mean")
            total_loss += loss.item() * im_hr.size(0)
            count += im_hr.size(0)
        self.model.train()
        return total_loss / count if count > 0 else 0.0

    @torch.no_grad()
    def sample_ddim(self, split="val"):
        sampler = DDIMScheduler(num_train_timesteps=1000)
        sampler.set_timesteps(50)
        self.model.eval()

        loader = self.val_loader if split == "val" else self.train_loader
        batch = next(iter(loader))
        im_lr, im_hr_vv, im_hr_vh, cond_angle, cond_lc, cond_dem = [t.to(self.device) for t in batch]
        im_hr = torch.cat([im_hr_vv, im_hr_vh], dim=1) 
        sanity_check(im_lr, im_hr)

        cond_lc_onehot = onehot_spatial(cond_lc, NUM_CLASSES, self.device)
        cond_emb = lr_to_embedding(im_lr)
        cond_dem_normed = (cond_dem - 200) / 100

        latents = torch.randn((im_lr.size(0), HR_CHANNELS, IMG_SIZE_HR, IMG_SIZE_HR), device=self.device)

        for t in tqdm(sampler.timesteps):
            latents_cond = torch.cat([latents, im_lr, cond_angle, cond_lc_onehot, cond_dem_normed], dim=1)
            noise_pred = self.model(sample=latents_cond, timestep=t, encoder_hidden_states=cond_emb)["sample"]
            latents = sampler.step(noise_pred, t, latents).prev_sample

        save_image(latents, os.path.join(self.outputs_dir, f"ddim_generated_{split}.png"),
                   normalize=True, value_range=(latents.min(), latents.max()))

        total_psnr = 0.0
        for i in range(min(im_lr.size(0), 4)):
            pred = latents[i].detach().cpu().numpy()
            gt = im_hr[i].detach().cpu().numpy()
            psnr_val = np.mean([
                compute_psnr(gt[c], pred[c], data_range=gt[c].ptp()) for c in range(HR_CHANNELS)
            ])
            total_psnr += psnr_val

            fig, axs = plt.subplots(1, 3, figsize=(9, 3))
            axs[0].imshow(im_lr[i].squeeze().detach().cpu(), cmap="gray")
            axs[0].set_title("LR")
            axs[0].axis("off")
            axs[1].imshow(gt[0], cmap="gray")
            axs[1].set_title("HR (GT VV)")
            axs[1].axis("off")
            axs[2].imshow(pred[0], cmap="gray")
            axs[2].set_title(f"SR (VV) PSNR: {psnr_val:.2f}dB")
            axs[2].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(self.outputs_dir, f"plot_{split}_{i}.png"))
            plt.close()

        avg_psnr = total_psnr / min(im_lr.size(0), 4)
        print(f"Average PSNR on {split} set: {avg_psnr:.2f}dB")


# ------------------------- entry point ------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_ckpt", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    trainer = DiffusionSRTrainer(resume_ckpt=args.resume_ckpt)
    trainer.run()
