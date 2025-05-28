import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision.models as models
from tqdm import tqdm
import lpips

# 设置随机种子确保可复现性
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = datasets.CelebA(
    root='./data',
    split='train',
    download=False,
    transform=transform,
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# 后续代码保持不变...


# DiT模型实现
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.attention = nn.MultiheadAttention(channels, 8, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.group_norm(x)
        x = x.view(b, c, h * w).transpose(1, 2)
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.transpose(1, 2).view(b, c, h, w)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1)
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.activation = nn.SiLU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x + residual


class DiT(nn.Module):
    def __init__(self, image_size=256, in_channels=3, hidden_channels=256,
                 num_blocks=4):
        super().__init__()
        self.image_size = image_size

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            *[ResidualBlock(hidden_channels, hidden_channels) for _ in
              range(num_blocks)],
            AttentionBlock(hidden_channels),
            *[ResidualBlock(hidden_channels, hidden_channels) for _ in
              range(num_blocks)],
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3,
                      stride=2, padding=1),
            *[ResidualBlock(hidden_channels * 2, hidden_channels * 2) for _ in
              range(num_blocks)],
            AttentionBlock(hidden_channels * 2),
            *[ResidualBlock(hidden_channels * 2, hidden_channels * 2) for _ in
              range(num_blocks)],
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3,
                      stride=2, padding=1),
            *[ResidualBlock(hidden_channels * 4, hidden_channels * 4) for _ in
              range(num_blocks)],
            AttentionBlock(hidden_channels * 4),
            *[ResidualBlock(hidden_channels * 4, hidden_channels * 4) for _ in
              range(num_blocks)]
        )

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            ResidualBlock(hidden_channels * 4, hidden_channels * 4),
            AttentionBlock(hidden_channels * 4),
            ResidualBlock(hidden_channels * 4, hidden_channels * 4)
        )

        # 解码器
        self.decoder = nn.Sequential(
            *[ResidualBlock(hidden_channels * 4, hidden_channels * 4) for _ in
              range(num_blocks)],
            AttentionBlock(hidden_channels * 4),
            *[ResidualBlock(hidden_channels * 4, hidden_channels * 4) for _ in
              range(num_blocks)],
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2,
                               kernel_size=4, stride=2, padding=1),
            *[ResidualBlock(hidden_channels * 2, hidden_channels * 2) for _ in
              range(num_blocks)],
            AttentionBlock(hidden_channels * 2),
            *[ResidualBlock(hidden_channels * 2, hidden_channels * 2) for _ in
              range(num_blocks)],
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels,
                               kernel_size=4, stride=2, padding=1),
            *[ResidualBlock(hidden_channels, hidden_channels) for _ in
              range(num_blocks)],
            AttentionBlock(hidden_channels),
            *[ResidualBlock(hidden_channels, hidden_channels) for _ in
              range(num_blocks)],
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x


# 初始化模型、优化器和损失函数
model = DiT().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()
lpips_loss = lpips.LPIPS(net='vgg').to(device)

# 创建保存目录
os.makedirs('generated_images', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)


# 训练函数
def train(epoch):
    model.train()
    train_loss = 0
    lpips_score = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

    for batch_idx, (data, _) in progress_bar:
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch = model(data)

        # 计算损失
        mse_loss = criterion(recon_batch, data)
        perceptual_loss = lpips_loss(recon_batch, data).mean()
        loss = mse_loss + 0.1 * perceptual_loss

        loss.backward()
        train_loss += loss.item()
        lpips_score += perceptual_loss.item()
        optimizer.step()

        progress_bar.set_description(
            f'Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} '
            f'({100. * batch_idx / len(dataloader):.0f}%)]')
        progress_bar.set_postfix(loss=loss.item() / len(data),
                                 lpips=perceptual_loss.item() / len(data))

    print(
        f'====> Epoch: {epoch} Average loss: {train_loss / len(dataloader):.4f}, '
        f'Average LPIPS: {lpips_score / len(dataloader):.4f}')

    # 保存生成的图像和模型检查点
    with torch.no_grad():
        sample = data[:8].to(device)
        recon_sample = model(sample)
        comparison = torch.cat([sample, recon_sample])
        save_image(comparison.cpu(),
                   f'generated_images/reconstruction_{epoch}.png', nrow=8)

    torch.save(model.state_dict(), f'checkpoints/dit_model_epoch_{epoch}.pth')


# 训练模型
num_epochs = 50
for epoch in range(1, num_epochs + 1):
    train(epoch)    