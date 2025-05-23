# Super Resolution for Remote Sensing Data
In this project, we investigate super resolution based on deep learning for
remote sensing satellite images. In particular, we develop a vision transformer
as well as a diffusion model to achieve this.

## Data Setup

Our dataset class `LRHRDataSet` expects the dataset to have to following file
structure:

```
sr_data/
├── LR/
│   ├── region1-SMAP/
│   │   ├── 20170522.tif
│   │   └── ...
│   ├── region2-SMAP/
│   │   └── ...
│   └── ...
├── HR/
│   └── VV/
│       ├── region1-VV/
│       │   ├── 20170522.tif
│       │   └── ...
│       └── ...
├── IncAngle/
│   ├── region1-Ang/
│   │   ├── 20170522.tif
│   │   └── ...
│   └── ...
├── EWC/
│   ├── EWC556region1.tif
│   ├── EWC556region2.tif
│   └── ...
└── DEM/
    ├── DEM556region1.tif
    ├── DEM556region2.tif
    └── ...
```

## Installation and Execution

To install the required dependencies, simply run:
```
pip install -r requirements.txt
```

To train the diffusion model, you can run:
```
python -m src.diffusion.train_diff
```

Similar, for inference:
```
python -m src.diffusion.infer
```
