from diffusers import UNet2DModel, DDIMScheduler
from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import os

# ========== 1. 加载模型和调度器 ==========
model = UNet2DModel.from_pretrained("google/ddpm-celebahq-256").cuda()
scheduler = DDIMScheduler.from_pretrained("google/ddpm-celebahq-256", subfolder="scheduler")
scheduler.set_timesteps(50)

# ========== 2. 加载图像并生成遮挡区域 ==========
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

image = Image.open("face_001.png").convert("RGB")
x_0 = transform(image).unsqueeze(0).cuda()  # shape: [1, 3, 256, 256]

# 创建中心方块掩码
mask = torch.ones_like(x_0)
mask[:, :, 96:160, 96:160] = 0  # 中间方块遮住

# 显示掩码图像
masked_image = (x_0 * mask).clamp(0, 1).squeeze().permute(1, 2, 0).cpu().numpy()
plt.figure(figsize=(6, 6))
plt.imshow(masked_image)
plt.title("Masked Input (Center Block)")
plt.axis("off")
plt.show()

# 添加随机噪声起点
x = torch.randn_like(x_0)

# ========== 3. RePaint 推理参数 ==========
jump_every = 10   # 每10步跳一次
jump_n = 1        # 跳回多少步
max_jumps = 5     # 最多跳几次
jump_counter = 0

# 为推理可视化准备保存目录
os.makedirs("inference_steps", exist_ok=True)
save_interval = 10  # 每隔多少步保存一次图像
step_images = []

# ========== 4. RePaint DDIM 跳步推理 ==========
for i, t in enumerate(scheduler.timesteps):
    # 当前步推理
    with torch.no_grad():
        noise_pred = model(x, t).sample
        x_prev = scheduler.step(noise_pred, t, x).prev_sample

    # Inpainting：保持非掩码区域不变
    x = x_prev * mask + x_0 * (1 - mask)

    # === RePaint跳步逻辑 ===
    if (i > 0) and (i % jump_every == 0) and (jump_counter < max_jumps):
        jump_counter += 1
        back_i = max(0, i - jump_n)
        t_back = scheduler.timesteps[back_i]

        # 回头一步
        with torch.no_grad():
            noise_pred = model(x, t_back).sample
            x_forward = scheduler.step(noise_pred, t_back, x).prev_sample

        # 再前向采样回来
        with torch.no_grad():
            noise_pred = model(x_forward, t).sample
            x = scheduler.step(noise_pred, t, x_forward).prev_sample

        x = x * mask + x_0 * (1 - mask)

    # 推理过程可视化保存
    if i % save_interval == 0 or i == scheduler.num_inference_steps - 1:
        vis = x.clamp(0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy()
        step_images.append(vis)

# ========== 5. 显示最终结果 ==========
recon = x.clamp(0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy()
orig = x_0.cpu().squeeze(0).permute(1, 2, 0).numpy()

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(orig)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(masked_image)
plt.title("Masked Input")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(recon)
plt.title("Inpainted (DDIM + RePaint)")
plt.axis("off")
plt.show()

# ========== 6. 显示推理过程图像序列 ==========
fig, axs = plt.subplots(1, len(step_images), figsize=(3 * len(step_images), 3))
for idx, img in enumerate(step_images):
    axs[idx].imshow(img)
    axs[idx].set_title(f"Step {idx * save_interval}")
    axs[idx].axis("off")
plt.suptitle("DDIM + RePaint Inference Progress")
plt.show()
