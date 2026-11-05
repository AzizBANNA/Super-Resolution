# Super-Resolution for Remote Sensing Data

Deep-learning super-resolution for satellite remote sensing images. A
conditional diffusion model (a diffusers U-Net, DDPM training with DDIM sampling)
takes a low-resolution SMAP image and predicts the high-resolution Sentinel-1
backscatter. It is conditioned on additional layers:

| Input            | Description                                   |
|------------------|-----------------------------------------------|
| LR               | Low-resolution SMAP image                     |
| DEM              | Digital elevation model                       |
| EWC (land cover) | Land cover classes, one-hot encoded (9 classes) |
| IncAngle         | Incidence angle (dual-channel model only)     |

There are two variants of the model:

- **Single-channel** (`train_diff.py` / `infer.py`): predicts HR **VV**.
- **Dual-channel** (`train_diff_2HR.py` / `infer_2HR.py`): predicts HR **VV + VH** together.

## Project Structure

```
src/
├── data/
│   ├── lrhr_dataset.py       # Dataset for the single-channel (VV) model
│   └── lrhr_dataset_2HR.py   # Dataset for the dual-channel (VV + VH) model
└── diffusion/
    ├── train_diff.py         # Train single-channel model
    ├── infer.py              # Inference + metrics (PSNR, SSIM, SCC, MSE) per region
    ├── train_diff_2HR.py     # Train dual-channel model
    └── infer_2HR.py          # Inference + metrics for dual-channel model
```

## Data Setup

The datasets expect this folder structure (the `VH` folder is needed only by
the dual-channel model):

```
sr_data/
├── LR/
│   ├── region1-SMAP/
│   │   ├── 20170522.tif
│   │   └── ...
│   └── ...
├── HR/
│   ├── VV/
│   │   ├── region1-VV/
│   │   │   ├── 20170522.tif
│   │   │   └── ...
│   │   └── ...
│   └── VH/
│       ├── region1-VH/
│       └── ...
├── IncAngle/
│   ├── region1-Ang/
│   │   ├── 20170522.tif
│   │   └── ...
│   └── ...
├── EWC/
│   ├── EWC556region1.tif
│   └── ...
└── DEM/
    ├── DEM556region1.tif
    └── ...
```

Files are matched across folders by their `YYYYMMDD` timestamp. All inputs are
z-score normalized with statistics from the training set.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run all commands from the repository root.

**Single-channel (VV)**: train and validate on separate folders (e.g. a
temporal split):

```bash
python -m src.diffusion.train_diff --train_root sr_data_train --val_root sr_data_val
python -m src.diffusion.infer --ckpt results/<run>/checkpoints/best.pt --data_root sr_data
```

**Dual-channel (VV + VH)**: random 90/10 train/val split of one folder:

```bash
python -m src.diffusion.train_diff_2HR --data_root sr_data
python -m src.diffusion.infer_2HR --ckpt results/<run>/checkpoints/best.pt --data_root sr_data
```

Training writes checkpoints, TensorBoard logs and sample plots to
`results/<timestamp>/`. Use `--resume_ckpt <path>` to continue a run.

## Credits

Developed together with Julian Schmitt and Yahya Hefnawi. Original repository:
[JulianPSchmitt/sr-for-remote-sensing](https://github.com/JulianPSchmitt/sr-for-remote-sensing).
