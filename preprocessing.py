import pandas as pd
import numpy as np
import random 
from pathlib import Path

train_dir = Path("train")

# One dict per location, holding every file for it, so they all stay together when split.
# Every file for a location starts with the same stem (ex: "guatemala-volcano_00000001"),
# so each path is built from that stem and can never be matched to the wrong location.
pairs = []
for before_image in sorted((train_dir / "images").glob("*_pre_disaster.png")):
    stem = before_image.name.replace("_pre_disaster.png", "")
    pairs.append({
        "disaster":     stem.split("_")[0],
        "before_image": before_image,
        "after_image":  train_dir / "images"  / f"{stem}_post_disaster.png",
        "before_mask":  train_dir / "targets" / f"{stem}_pre_disaster_target.png",
        "after_mask":   train_dir / "targets" / f"{stem}_post_disaster_target.png",
        "before_label": train_dir / "labels"  / f"{stem}_pre_disaster.json",
        "after_label":  train_dir / "labels"  / f"{stem}_post_disaster.json",
        # Uncompressed copies of the two images, made once by convert_to_npy.py.
        # BuildingCropDataset memory maps these to read only the rows a crop needs.
        "before_npy":   train_dir / "npy"     / f"{stem}_pre_disaster.npy",
        "after_npy":    train_dir / "npy"     / f"{stem}_post_disaster.npy",
    })

# Make sure every location actually has all of its files.
# The .npy files are skipped here, they don't exist until convert_to_npy.py runs,
# and that script imports this file. BuildingCropDataset checks for them instead.
NOT_CHECKED = {"disaster", "before_npy", "after_npy"}
for pair in pairs:
    for key, path in pair.items():
        if key not in NOT_CHECKED:
            assert path.exists(), f"Missing {key}: {path}"


if __name__ == "__main__":

    # Every before image should have a matching after image, two masks, and two label files
    print("Locations: " + str(len(pairs)))
    print("Images: " + str(len(list((train_dir / "images").glob("*.png")))))
    print("Pixel Masks: " + str(len(list((train_dir / "targets").glob("*.png")))))
    print("Labels: " + str(len(list((train_dir / "labels").glob("*.json")))))

    # Shuffle with a fixed seed so the split is the same every run, then take out 15% for validation
    random.seed(42)
    random.shuffle(pairs)
    num_val = int(0.15 * len(pairs))
    val_pairs = pairs[:num_val]
    train_pairs = pairs[num_val:]

    print("Train Pairs: " + str(len(train_pairs)))
    print("Validation Pairs: " + str(len(val_pairs)))


