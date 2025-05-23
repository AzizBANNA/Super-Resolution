# Diffusion-based RGB super-resolution trainer with DDIM sampling

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from accelerate import Accelerator
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from src.data.lrhr_dataset import LRHRDataSet
from torchvision.utils import save_image
import os

# ------------------------- hyperparameters ------------------------- #
NUM_EPOCHS = 500
SAVE_FREQ = 50
BATCH_SIZE = 2
LR = 1e-4
LOG_PATH = "diff_logs"
CKPT_DIR = "results/diffusion_baseline_checkpoints"
IMG_SIZE_HR = 256
USE_MIXED_PREC = True


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
    if im_hr.shape[1] != 1 or im_lr.shape[1] != 1:
        raise ValueError("Expected 1-channel grayscale inputs")
    if im_hr.shape[-2:] != (256, 256):
        raise ValueError(f"HR image size is not 256x256 but {im_hr.shape[-2:]}")
    if torch.isnan(im_lr).any() or torch.isnan(im_hr).any():
        raise ValueError("Input contains NaN values")


# ------------------------- trainer ------------------------- #
class DiffusionSRTrainer:
    def __init__(self):
        self.acc = build_accelerator()
        self.device = self.acc.device

        # train_ds = LRHRDataSet(mode="train", indices=np.arange(0, 120))
        # val_ds = LRHRDataSet(mode="train", indices=np.arange(120, 142))
        train_ds = LRHRDataSet(mode="train", indices=np.arange(0, 30))
        val_ds = LRHRDataSet(mode="train", indices=np.arange(30, 35))
        self.train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
        )

        self.model = UNet2DConditionModel(
            sample_size=IMG_SIZE_HR,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 256),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
            class_embed_type=None,
            encoder_hid_dim=1,
        )

        self.scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.opt = optim.AdamW(self.model.parameters(), lr=LR)
        self.writer = SummaryWriter(LOG_PATH)
        self.step = self.epoch = 0

        self.model, self.opt, self.train_loader, self.val_loader = self.acc.prepare(
            self.model, self.opt, self.train_loader, self.val_loader
        )

    def train_step(self, batch):
        im_lr, im_hr, _, _, _ = batch
        im_lr = im_lr.to(self.device).float()
        im_hr = im_hr.to(self.device).float()

        sanity_check(im_lr, im_hr)

        b = im_hr.size(0)
        t = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (b,), device=self.device
        ).long()
        noise = torch.randn_like(im_hr)
        x_t = self.scheduler.add_noise(im_hr, noise, t)

        cond_img = im_lr
        cond_emb = lr_to_embedding(cond_img)

        with self.acc.autocast():
            pred_noise = self.model(x_t, t, encoder_hidden_states=cond_emb)["sample"]
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
            if (self.epoch + 1) % SAVE_FREQ == 0:
                self.save_ckpt()
        self.sample_ddim(split="val")

    def save_ckpt(self):
        Path(CKPT_DIR).mkdir(exist_ok=True)
        torch.save(
            {
                "model": self.acc.get_state_dict(self.model),
                "opt": self.opt.state_dict(),
                "epoch": self.epoch,
                "step": self.step,
            },
            Path(CKPT_DIR) / f"diff_sr_ep{self.epoch}.pt",
        )

    @torch.no_grad()
    def sample_ddim(self, split="val"):
        sampler = DDIMScheduler(num_train_timesteps=1000)
        sampler.set_timesteps(50)
        self.model.eval()

        loader = self.val_loader if split == "val" else self.train_loader
        batch = next(iter(loader))
        im_lr, im_hr, _, _, _ = [t.to(self.device) for t in batch]
        sanity_check(im_lr, im_hr)

        # cond_img = upsample_to_hr(im_lr, (IMG_SIZE_HR, IMG_SIZE_HR))
        cond_emb = lr_to_embedding(im_lr)

        latents = torch.randn(
            (im_lr.size(0), 1, IMG_SIZE_HR, IMG_SIZE_HR), device=self.device
        )

        for t in tqdm(sampler.timesteps):
            noise_pred = self.model(
                sample=latents, timestep=t, encoder_hidden_states=cond_emb
            )["sample"]
            latents = sampler.step(noise_pred, t, latents).prev_sample

        os.makedirs("outputs_baseline", exist_ok=True)
        save_image(
            latents,
            f"outputs_baseline/ddim_generated_{split}.png",
            normalize=True,
            value_range=(0, 1),
        )

        total_psnr = 0.0
        for i in range(min(im_lr.size(0), 4)):
            pred = latents[i].detach().cpu().permute(1, 2, 0).numpy()
            gt = im_hr[i].detach().cpu().permute(1, 2, 0).numpy()
            psnr_val = compute_psnr(gt, pred, data_range=gt.max() - gt.min())
            total_psnr += psnr_val

            fig, axs = plt.subplots(1, 3, figsize=(9, 3))
            axs[0].imshow(im_lr[i].detach().cpu().permute(1, 2, 0).numpy())
            axs[0].set_title("LR")
            axs[0].axis("off")

            axs[1].imshow(gt)
            axs[1].set_title("HR (GT)")
            axs[1].axis("off")

            axs[2].imshow(pred)
            axs[2].set_title(f"SR (PSNR: {psnr_val:.2f}dB)")
            axs[2].axis("off")

            plt.tight_layout()
            plt.savefig(f"outputs_baseline/plot_{split}_{i}.png")
            plt.close()

        avg_psnr = total_psnr / min(im_lr.size(0), 4)
        print(f"Average PSNR on {split} set: {avg_psnr:.2f}dB")


if __name__ == "__main__":
    trainer = DiffusionSRTrainer()
    trainer.run()
