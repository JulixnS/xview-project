import pandas as pd
import numpy as np
import random 
from pathlib import Path

train_dir = Path("train")
before_images = sorted((train_dir / "images").glob("*pre_disaster.png"))
after_images = sorted((train_dir / "images").glob("*post_disaster.png"))
before_pixel_masks = sorted((train_dir / "targets").glob("*pre_disaster_target.png"))
after_pixel_masks = sorted((train_dir / "targets").glob("*post_disaster_target.png"))

# These should all be the same number, aka every pre image has a post image, and every image has a corresponding mask
print("Before Images: " + str(len(before_images)))
print("After Images: " + str(len(after_images)))
print("Before Pixel Masks: " + str(len(before_pixel_masks)))
print("After Pixel Masks: " + str(len(after_pixel_masks)))

# Group each before image with its after image and both masks, so they stay together when split
pairs = list(zip(before_images, after_images, before_pixel_masks, after_pixel_masks))

# zip pairs by position only, so make sure every entry is actually the same location
for before_image, after_image, before_pixel_mask, after_pixel_mask in pairs:
    stem = before_image.name.replace("_pre_disaster.png", "")
    assert after_image.name == stem + "_post_disaster.png", (before_image, after_image)
    assert before_pixel_mask.name == stem + "_pre_disaster_target.png", (before_image, before_pixel_mask)
    assert after_pixel_mask.name == stem + "_post_disaster_target.png", (before_image, after_pixel_mask)

# Shuffle with a fixed seed so the split is the same every run, then take out 15% for validation
random.seed(42)
random.shuffle(pairs)
num_val = int(0.15 * len(pairs))
val_pairs = pairs[:num_val]
train_pairs = pairs[num_val:]

print("Train Pairs: " + str(len(train_pairs)))
print("Validation Pairs: " + str(len(val_pairs)))


