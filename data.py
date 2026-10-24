import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2 
from preprocessing import pairs

class BuildingDataset(Dataset):
    def __init__(self, pairs, transform = None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        before_image, after_image, before_pixel_mask, after_pixel_mask = self.pairs[index]
        img = np.array(Image.open(before_image).convert("RGB"))
        mask = np.array(Image.open(before_pixel_mask))
        output = self.transform(image=img, mask=mask)

        return output["image"], output["mask"].unsqueeze(0).float()
        

if __name__ == "__main__":
    normalize = A.Compose([
        A.Normalize(mean = (0.485, 0.456, 0.406),  std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]) 
    identity = A.Compose([ToTensorV2()])  #Applies no transformation, just turns into a tensor
    
    normalized_images = BuildingDataset(pairs, normalize)
    raw_images = BuildingDataset(pairs, identity)

    print(normalized_images[0])
    print(raw_images[0])

