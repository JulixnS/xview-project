import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import albumentations as A
from albumentations.pytorch import ToTensorV2
from preprocessing import pairs
from abc import ABC, abstractmethod

# Damage level names in the label JSONs
# "un-classified" is left out on purpose, those buildings have no usable damage label.
DAMAGE_CLASSES = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}

# ===== Helper Funtions =====

def load_image(path):
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path):
    return np.array(Image.open(path))

# Turn a WKT string like "POLYGON ((x1 y1, x2 y2, ...))" into an (N, 2) array of x, y points.
def parse_wkt_polygon(wkt):
    inside = wkt[wkt.index("((") + 2 : wkt.index(")")]
    points = [point.split() for point in inside.split(",")]
    return np.array(points, dtype=np.float32)


def load_buildings(json_path):
    """
    Read a label JSON and return one dict per building:
        "uid":     the building's id, the same building has the same uid in the pre and post JSON
        "polygon": (N, 2) array of pixel x, y points outlining the building
        "damage":  class index from DAMAGE_CLASSES, or None for pre disaster labels and un-classified buildings
    """
    with open(json_path) as f:
        label = json.load(f)

    buildings = []
    for feature in label["features"]["xy"]:   #features[xy] is a list of all the buildings
        properties = feature["properties"]
        buildings.append({
            "uid": properties["uid"],
            "polygon": parse_wkt_polygon(feature["wkt"]),
            "damage": DAMAGE_CLASSES.get(properties.get("subtype")),
        })
    return buildings


def polygon_to_bbox(polygon):
    """Smallest box around a polygon, returned as (x_min, y_min, x_max, y_max)."""
    x_min, y_min = polygon.min(axis=0)
    x_max, y_max = polygon.max(axis=0)
    return float(x_min), float(y_min), float(x_max), float(y_max)


def polygon_to_mask(polygon, height, width):
    """Fill in a polygon on an empty (height, width) canvas. Building pixels are 1, everything else is 0."""
    canvas = Image.new("L", (width, height), 0)
    ImageDraw.Draw(canvas).polygon([tuple(point) for point in polygon], fill=1)
    return np.array(canvas)

# ===== Dataset Classes =====

class ABCDataset(ABC, Dataset):
    def __init__(self, pairs, transform = None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    @abstractmethod
    def __getitem__(self, index):
        ...

class SegmentationDataset(ABCDataset):
    def __init__(self, pairs, transform = None):
        if transform is None:
            transform = A.Compose([A.Normalize(), ToTensorV2()])
        super().__init__(pairs, transform)

    def __getitem__(self, index):
        #The building locator model only trains on before images, the target is the before_pixel mask. After images and masks are discarded
        pair = self.pairs[index]
        img = load_image(pair["before_image"])
        mask = load_mask(pair["before_mask"])
        output = self.transform(image=img, mask=mask)

        return output["image"], output["mask"].unsqueeze(0).float()

class InstanceDataset(ABCDataset):
    def __getitem__(self, index):
        pair = self.pairs[index]
        img = load_image(pair["before_image"])
        height, width = img.shape[:2]
        buildings = load_buildings(pair["before_label"])

        # Clip each box to the image edges, buildings cut off at the border can have corners outside it.
        # Then drop any box under 1 pixel wide or tall, Mask R-CNN errors on zero width boxes.
        # Filtering the buildings themselves keeps boxes, labels, and masks lined up.
        kept = []
        for b in buildings:
            x_min, y_min, x_max, y_max = polygon_to_bbox(b["polygon"])
            x_min, x_max = np.clip([x_min, x_max], 0, width)
            y_min, y_max = np.clip([y_min, y_max], 0, height)
            if x_max - x_min >= 1 and y_max - y_min >= 1:
                kept.append((b, (x_min, y_min, x_max, y_max)))

        boxes = [box for _, box in kept]
        boxes = torch.tensor(boxes, dtype = torch.float32).reshape(-1, 4)
        labels = torch.ones(len(kept), dtype=torch.int64)
    
        masks = np.zeros((len(kept), height, width), dtype=np.uint8)
        for k, (b, _) in enumerate(kept):
            masks[k] = polygon_to_mask(b["polygon"], height, width)
        masks = torch.from_numpy(masks)

        image_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        target = {"boxes": boxes, "labels": labels, "masks": masks}
        return image_tensor, target

class BuildingCropDataset(ABCDataset):
    def __init__(self, pairs, transform = None):
        if transform is None:
            transform = A.Compose(
                [A.Resize(128, 128), A.Normalize(), ToTensorV2()],
                additional_targets={"after": "image"},
            )
        super().__init__(pairs, transform)

        # Crops are read from the .npy copies of the images, made once by convert_to_npy.py.
        # Check up front so a missing file fails here, not partway through training.
        missing = [pair[key] for pair in pairs for key in ("before_npy", "after_npy") if not pair[key].exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} .npy files are missing (first one: {missing[0]}). "
                "Run python convert_to_npy.py first."
            )

        self.samples = []
        for pair in pairs:
            for b in load_buildings(pair["after_label"]):
                if b["damage"] is None:  #skip un-classified buildings
                    continue

                box = polygon_to_bbox(b["polygon"])
                self.samples.append({
                    "pair": pair,
                    "box": box,
                    "damage": b["damage"]
                })

    def __len__(self):
        return len(self.samples)

    def _crop_square(self, box, height, width, pad=0.25, min_side=32):
        x_min, y_min, x_max, y_max = box
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2       # center of the building
        side = max(x_max - x_min, y_max - y_min) * (1 + 2 * pad)  # longest side plus padding
        side = int(min(max(side, min_side), height, width))       # at least min_side, at most the image

        x0 = int(round(cx - side / 2))
        y0 = int(round(cy - side / 2))
        x0 = min(max(x0, 0), width - side)                        # shift back inside the image
        y0 = min(max(y0, 0), height - side)
        return x0, y0, x0 + side, y0 + side

    def __getitem__(self, index):
        sample = self.samples[index]

        # mmap_mode="r" opens the file without reading the pixels. Only the rows the
        # crop slice touches get read from disk, instead of decoding the whole PNG.
        before = np.load(sample["pair"]["before_npy"], mmap_mode="r")
        after = np.load(sample["pair"]["after_npy"], mmap_mode="r")

        height, width = before.shape[:2]   # comes from the file header, no pixels read
        x0,y0,x1,y1 = self._crop_square(sample["box"], height, width)

        # The slice skips the rest of each row, so it isn't one solid block of memory.
        # Copy it into one (only a few KB) so the resize in the transform can use it.
        before_crop = np.ascontiguousarray(before[y0:y1, x0:x1])
        after_crop = np.ascontiguousarray(after[y0:y1, x0:x1])

        out = self.transform(image=before_crop, after=after_crop)
        return out["image"], out["after"], sample["damage"]



            
        

if __name__ == "__main__":
    normalize = A.Compose([
        A.Normalize(mean = (0.485, 0.456, 0.406),  std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]) 
    identity = A.Compose([ToTensorV2()])  #Applies no transformation, just turns into a tensor
    
    normalized_images = SegmentationDataset(pairs, normalize)
    raw_images = SegmentationDataset(pairs, identity)

    print(normalized_images[0])
    print(raw_images[0])

