import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from einops import rearrange, repeat


# 基础模块：LayerNorm和残差连接
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


# 时间嵌入模块
class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        self.emb = nn.Embedding(1000, half_dim)
        self.proj = nn.Linear(half_dim, dim)

    def forward(self, t):
        t = self.emb(t)
        t = F.silu(t)
        t = self.proj(t)
        return t


# Transformer模块
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), LayerNorm(
            dim)) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, heads=heads, dim_head=dim_head)
        self.norm2 = LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# DiT模型主体
class DiT(nn.Module):
    def __init__(self, image_size=256, in_channels=3, dim=512, depth=6, heads=8,
                 dim_head=64):
        super().__init__()
        self.image_size = image_size
        self.patch_size = 16
        self.patch_dim = in_channels * (patch_size ** 2)
        self.num_patches = (image_size // patch_size) ** 2

        # 图像到补丁的映射
        self.patch_proj = nn.Linear(self.patch_dim, dim)

        # 位置嵌入
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # 时间嵌入
        self.time_emb = TimestepEmbedding(dim)

        # Transformer块
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, heads=heads, dim_head=dim_head)
            for _ in range(depth)
        ])

        # 输出层
        self.to_out = nn.Linear(dim, self.patch_dim)

    def forward(self, x, t):
        b, c, h, w = x.shape
        assert h == w == self.image_size, f"图像尺寸应为{self.image_size}x{self.image_size}"

        # 补丁化
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)',
                      p1=self.patch_size, p2=self.patch_size)
        x = self.patch_proj(x)

        # 添加类令牌
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # 添加位置嵌入
        x = x + self.pos_emb

        # 添加时间嵌入
        t_emb = self.time_emb(t)
        t_emb = rearrange(t_emb, 'b d -> b 1 d')
        x = x + t_emb

        # 通过Transformer块
        for block in self.transformer_blocks:
            x = block(x)

        # 提取补丁表示（不包括类令牌）
        x = x[:, 1:, :]

        # 映射回补丁空间
        x = self.to_out(x)

        # 重塑回图像形状
        x = rearrange(x, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                      h=self.image_size // self.patch_size,
                      w=self.image_size // self.patch_size,
                      p1=self.patch_size, p2=self.patch_size, c=c)

        return x


# 扩散过程
class Diffusion:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0),
                                         value=1.0)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (self.alphas_cumprod[t][:, None, None, None] ** 0.5) * x_start + \
            (1 - self.alphas_cumprod[t][:, None, None, None]) ** 0.5 * noise

    def get_loss(self, model, x_start, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        x_pred = model(x_noisy, t)
        loss = F.mse_loss(x_pred, noise, reduction='none').mean()
        return loss


# 训练函数
def train_dit(model, diffusion, train_loader, optimizer, epochs=100,
              device="cuda"):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in progress_bar:
            batch = batch.to(device)
            optimizer.zero_grad()

            # 随机时间步
            t = torch.randint(0, diffusion.timesteps, (batch.shape[0],),
                              device=device).long()

            # 计算损失
            loss = diffusion.get_loss(model, batch, t)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

        # 定期保存模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"dit_model_epoch_{epoch + 1}.pt")


# 初始化模型、扩散过程和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DiT(image_size=256, in_channels=3, dim=512, depth=6)
diffusion = Diffusion(timesteps=1000)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 开始训练
train_dit(model, diffusion, train_loader, optimizer, epochs=50)