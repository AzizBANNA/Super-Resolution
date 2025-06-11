import os
import re
import rasterio
import numpy as np
import cv2
import random
from torch.utils.data import Dataset

random.seed(42)
np.random.seed(42)


class LRHRDataSet(Dataset):
    def __init__(
        self, data_root="sr_data", mode="train", indices=None, zscore=True, augment=True
    ):
        self.mode = mode
        self.lr_root = os.path.join(data_root, "LR")
        self.hr_root = os.path.join(data_root, "HR", "VV")
        self.angle_root = os.path.join(data_root, "IncAngle")
        self.ewc_root = os.path.join(data_root, "EWC")
        self.dem_root = os.path.join(data_root, "DEM")
        self.zscore = zscore
        self.augment = augment

        self.samples = []  # list of (lr_path, hr_path, angle_path, lc_path, dem_path)
        self.build_samples()
        if indices is None:
            self.samples = np.array(self.samples)
        else:
            self.samples = np.array(self.samples)[indices]

        # Compute mean and std for each channel if zscore normalization is enabled
        if self.zscore:
            self._compute_zscore_stats()

    def extract_timestamp(self, filename):
        """Extract date string from filename (format: 20170522 or similar)."""
        match = re.search(r"(20\d{6})", filename)
        return match.group(1) if match else None

    def build_samples(self):
        for region_folder in sorted(os.listdir(self.lr_root)):
            # Skip if it's not a directory
            if not os.path.isdir(os.path.join(self.lr_root, region_folder)):
                continue
            region = region_folder.replace("-SMAP", "")

            lr_dir = os.path.join(self.lr_root, region_folder)
            hr_dir = os.path.join(self.hr_root, f"{region}-VV")
            angle_dir = os.path.join(self.angle_root, f"{region}-Ang")
            ewc_path = os.path.join(self.ewc_root, f"EWC556{region}.tif")
            dem_path = os.path.join(self.dem_root, f"DEM556{region}.tif")

            for lr_file in sorted(os.listdir(lr_dir)):
                timestamp = self.extract_timestamp(lr_file)
                if timestamp is None:
                    continue

                hr_file = self.find_matching_file(hr_dir, timestamp)
                angle_file = self.find_matching_file(angle_dir, timestamp)

                if hr_file and angle_file:
                    sample = (
                        os.path.join(lr_dir, lr_file),
                        os.path.join(hr_dir, hr_file),
                        os.path.join(angle_dir, angle_file),
                        ewc_path,
                        dem_path,
                    )
                    self.samples.append(sample)

    def find_matching_file(self, folder, timestamp):
        """Find the first file in the folder containing the given timestamp."""
        for fname in os.listdir(folder):
            if timestamp in fname:
                return fname
        return None

    def _compute_zscore_stats(self):
        # Compute mean and std for each modality over the dataset, ignoring NaNs
        means = [[] for _ in range(5)]
        stds = [[] for _ in range(5)]
        for idx in range(len(self.samples)):
            paths = self.samples[idx]
            ims = []
            for p in paths:
                arr = rasterio.open(p).read().squeeze().astype(np.float32)
                arr = arr[np.isfinite(arr)]
                ims.append(arr)
            for i, arr in enumerate(ims):
                if arr.size > 0:
                    means[i].append(arr.mean())
                    stds[i].append(arr.std())
        # Store global mean and std for each modality
        self._means = [float(np.mean(m)) if len(m) > 0 else 0.0 for m in means]
        self._stds = [float(np.mean(s)) if len(s) > 0 else 1.0 for s in stds]

    def zscore_normalize(self, img, idx):
        img = np.nan_to_num(img)  # Replace NaNs with zero before normalization
        mean = self._means[idx]
        std = self._stds[idx]
        if std == 0:
            return img - mean
        return (img - mean) / std

    def _random_flip_and_rotate(self, ims):
        # ims: list of np arrays, all shape (H, W)
        # Random horizontal flip
        if random.random() > 0.5:
            ims = [np.fliplr(im) for im in ims]
        # Random vertical flip
        if random.random() > 0.5:
            ims = [np.flipud(im) for im in ims]
        # Random 90-degree rotation
        k = random.randint(0, 3)
        if k > 0:
            ims = [np.rot90(im, k) for im in ims]
        return ims

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lr_path, hr_path, angle_path, lc_path, dem_path = self.samples[idx]

        with rasterio.open(lr_path) as src:
            im_lr = src.read().squeeze().astype(np.float32)
        with rasterio.open(hr_path) as src:
            im_hr = src.read().squeeze().astype(np.float32)
        with rasterio.open(angle_path) as src:
            im_angle = src.read().squeeze().astype(np.float32)
        with rasterio.open(lc_path) as src:
            im_lc = src.read().squeeze().astype(np.float32)
        with rasterio.open(dem_path) as src:
            im_dem = src.read().squeeze().astype(np.float32)

        # Crop and resize to uniform size
        im_lr, im_hr, im_angle, im_lc, im_dem = [
            cv2.resize(im[3:256, 2:253], (256, 256), interpolation=cv2.INTER_NEAREST)
            for im in [im_lr, im_hr, im_angle, im_lc, im_dem]
        ]

        # --- Data augmentation (only in training mode) ---
        if self.mode == "train" and self.augment:
            im_lr, im_hr, im_angle, im_lc, im_dem = self._random_flip_and_rotate(
                [im_lr, im_hr, im_angle, im_lc, im_dem]
            )

        if self.zscore:
            im_lr = self.zscore_normalize(im_lr, 0)
            im_hr = self.zscore_normalize(im_hr, 1)
            im_angle = self.zscore_normalize(im_angle, 2)
            im_lc = self.zscore_normalize(im_lc, 3)
            im_dem = self.zscore_normalize(im_dem, 4)

        return im_lr[None], im_hr[None], im_angle[None], im_lc[None], im_dem[None]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    dataset = LRHRDataSet()
    os.makedirs("sanity_plots", exist_ok=True)

    for i in range(min(10, len(dataset))):  # Check the first 10 samples
        im_lr, im_hr, im_angle, im_lc, im_dem = dataset[i]

        images = {
            "LR": im_lr.squeeze(),
            "HR": im_hr.squeeze(),
            "Angle": im_angle.squeeze(),
            "Land Cover": im_lc.squeeze(),
            "DEM": im_dem.squeeze(),
        }

        print(f"\nSample {i}:")
        for name, img in images.items():
            min_val = np.nanmin(img)
            max_val = np.nanmax(img)
            nan_count = np.isnan(img).sum()
            print(
                f"  {name:<12} | Min: {min_val:.3f} | Max: {max_val:.3f} | NaNs: {nan_count}"
            )

        # Optional: replace NaNs with 0 for visualization if needed
        for key in images:
            if np.isnan(images[key]).any():
                images[key] = np.nan_to_num(images[key])

        # Plot
        plt.figure(figsize=(15, 3))
        for j, (name, img) in enumerate(images.items()):
            plt.subplot(1, 5, j + 1)
            if name in ["LR", "HR"]:
                cmap = "gray"
            elif name == "Angle":
                cmap = "viridis"  # or 'plasma', 'cividis', etc.
            elif name == "Land Cover":
                cmap = "tab20"
            else:  # DEM
                cmap = "terrain"
            plt.imshow(img, cmap=cmap)
            plt.title(name)
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"sanity_plots/sample_{i}.png")
        plt.close()
