"""
One time conversion of every before and after image from PNG to .npy.

PNG is compressed, so reading any part of an image means decoding the whole thing.
A .npy file is the raw pixel array, so BuildingCropDataset can memory map it and read
only the rows a building crop needs (about 150x faster per building).

Run once from the project root before training the Part 2 models:
    python convert_to_npy.py             # convert everything (about 18 GB, a few minutes)
    python convert_to_npy.py --limit 5   # only the first 5 locations, for testing

Safe to rerun. Files that already exist and have the right shape are skipped,
so a run that gets cut off just picks up where it stopped.
"""
import argparse
import os
import shutil
import numpy as np
from preprocessing import pairs
from data import load_image

BYTES_PER_IMAGE = 1024 * 1024 * 3 + 128   # raw uint8 pixels plus the .npy header


def is_valid(npy_path, expected_shape):
    """True if the .npy file exists and holds a uint8 array of the expected shape."""
    if not npy_path.exists():
        return False
    try:
        array = np.load(npy_path, mmap_mode="r")   # only reads the header
        return array.shape == expected_shape and array.dtype == np.uint8
    except (ValueError, OSError):
        return False   # half written or corrupt file


def convert(png_path, npy_path):
    """Decode one PNG and save it as .npy, then check the saved file."""
    image = load_image(png_path)

    # Write to a temporary name first, then rename. A rename is all or nothing,
    # so a job killed mid write never leaves a broken file under the real name.
    tmp_path = npy_path.with_suffix(".npy.tmp")
    with open(tmp_path, "wb") as f:
        np.save(f, image)
    os.replace(tmp_path, npy_path)

    if not is_valid(npy_path, image.shape):
        raise RuntimeError(f"Saved file failed its check: {npy_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert before/after PNG images to .npy")
    parser.add_argument("--limit", type=int, default=None, help="only convert the first N locations")
    args = parser.parse_args()

    selected = pairs[:args.limit] if args.limit else pairs
    jobs = [(pair[png_key], pair[npy_key])
            for pair in selected
            for png_key, npy_key in (("before_image", "before_npy"), ("after_image", "after_npy"))]

    npy_dir = jobs[0][1].parent
    npy_dir.mkdir(parents=True, exist_ok=True)

    # Skip anything already converted, then make sure the rest will fit on disk before starting
    todo = [(png, npy) for png, npy in jobs if not is_valid(npy, (1024, 1024, 3))]
    needed = len(todo) * BYTES_PER_IMAGE
    free = shutil.disk_usage(npy_dir).free
    print(f"{len(jobs)} images, {len(jobs) - len(todo)} already converted, {len(todo)} to go "
          f"({needed / 1e9:.1f} GB needed, {free / 1e9:.1f} GB free in {npy_dir})")
    if needed > free:
        raise SystemExit("Not enough disk space, stopping before writing anything.")

    for i, (png_path, npy_path) in enumerate(todo, start=1):
        convert(png_path, npy_path)
        if i % 500 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} converted")

    print("Done.")
