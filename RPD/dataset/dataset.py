from torch.utils.data.dataset import Dataset
from PIL import Image
import os
import cv2
from torchvision.transforms import ToTensor
import torch


class BUSIDataset_image_roi(Dataset):
    def __init__(self, image_path, mask_path, mapping_path, transform=None):
        super().__init__()
        self.image_path = image_path
        self.mask_path = mask_path
        self.mapping_path = mapping_path
        self.transform = transform

        self.image_names = []
        self.labels = []
        label_transform = {
                "Benign": 0,  # 良性
                "Malignant": 1,  # 恶性
            }
        
        with open(self.mapping_path, "r") as f:
            info = f.read().strip().split("\n")
            for line in info:
                name, label, diagnosis = line.split()
                self.image_names.append(name)
                self.labels.append(label_transform[label])

    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, index):
        image_name = self.image_names[index]
        label = self.labels[index]
        img = cv2.imread(self.image_path + image_name, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + image_name, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, img.shape[:2][::-1])

        T = self.transform(image=img, mask=mask)
        img, mask = T["image"], T["mask"]

        img = ToTensor()(Image.fromarray(img).convert("RGB"))
        mask = ToTensor()(Image.fromarray(mask).convert("L"))

        return img, mask, label
