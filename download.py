from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# 加载CelebA-HQ数据集
ds = load_dataset("bitmind/celeb-a-hq")

# 数据预处理：调整大小、归一化等
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # DiT通常处理256x256或更高分辨率的图像
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# 定义数据集类，应用预处理
class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset["train"])

    def __getitem__(self, idx):
        image = self.dataset["train"][idx]["image"]
        if self.transform:
            image = self.transform(image)
        return image


# 创建数据加载器
train_dataset = CelebADataset(ds, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                          num_workers=4)