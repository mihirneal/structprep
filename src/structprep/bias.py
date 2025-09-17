from pathlib import Path
import time
import json

import SimpleITK as sitk


def n4_bias_correct(in_file: Path, out_dir: Path, dry_run: bool = False) -> Path:
    out_path = out_dir / in_file.name.replace(".nii.gz", "_desc-biascorr.nii.gz")
    if dry_run:
        return out_path

    start = time.time()
    img = sitk.ReadImage(str(in_file))
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    out = corrector.Execute(img, mask)
    sitk.WriteImage(out, str(out_path))

    # Write a lightweight JSON log for N4
    meta = {
        "n4_applied": True,
        "method": "SimpleITK.N4BiasFieldCorrection",
        "elapsed_sec": round(time.time() - start, 3),
        "input_path": str(in_file),
        "output_path": str(out_path),
        "mask_method": "OtsuThreshold(levels=200)",
        "image_size": list(img.GetSize()),
        "image_spacing": list(img.GetSpacing()),
    }
    with open(str(out_path).replace(".nii.gz", ".json"), "w") as f:
        json.dump(meta, f, indent=2)

    return out_path
