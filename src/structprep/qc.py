from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def save_mosaic(in_file: Path, out_dir: Path, n_slices: int = 12):
    img = nib.load(str(in_file))
    data = img.get_fdata()
    z_slices = np.linspace(0, data.shape[2] - 1, n_slices, dtype=int)
    ncols = 6
    nrows = int(np.ceil(n_slices / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = axes.ravel()
    for i, z in enumerate(z_slices):
        axes[i].imshow(np.rot90(data[:, :, z]), cmap="gray")
        axes[i].axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    out_png = out_dir / in_file.name.replace(".nii.gz", "_desc-qc_mosaic.png")
    fig.savefig(out_png, dpi=100)
    plt.close(fig)

