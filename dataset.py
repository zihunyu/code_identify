import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CaptchaDataset(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        label = [int(c) for c in fname.split('.')[0]]  # 从文件名提取标签
        img = Image.open(os.path.join(self.root, fname)).convert('L')
        img = img.resize((100, 40))  # 与模型匹配
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.tensor(img).unsqueeze(0)  # shape: (1, H, W)
        return img, torch.tensor(label)
