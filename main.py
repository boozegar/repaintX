from diffusers import DDIMPipeline
from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import os

# ========== 1. 加载预训练 DDIMPipeline ==========
pipe = DDIMPipeline.from_pretrained("google/ddpm-celebahq-256").to("cuda")
scheduler = pipe.scheduler
scheduler.set_timesteps(50)

# ========== 2. 加载图像并生成遮挡区域 ==========
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

image = Image.open("face_001.png").convert("RGB")
x_0 = transform(image).unsqueeze(0).cuda()  # shape: [1, 3, 256, 256]

# 中心遮挡
mask = torch.ones_like(x_0)
mask[:, :, 96:160, 96:160] = 0  # 中间64x64区域遮挡
masked_input = x_0 * mask

# 起始噪声图像
x = torch.randn_like(x_0)

# 显示掩码图像
masked_image = masked_input.squeeze().permute(1, 2, 0).clamp(0, 1).cpu().numpy()
plt.imshow(masked_image)
plt.title("Masked Input")
plt.axis("off")
plt.show()

# ========== 3. RePaint 推理参数 ==========
jump_every = 10
jump_n = 1
max_jumps = 5
jump_counter = 0

# 可视化准备
os.makedirs("inference_steps", exist_ok=True)
step_images = []
save_interval = 10

# ========== 4. 使用 pipeline 的模型手动跳步生成 ==========
for i, t in enumerate(scheduler.timesteps):
    # 当前一步
    with torch.no_grad():
        noise_pred = pipe.unet(x, t).sample
        x_prev = scheduler.step(noise_pred, t, x).prev_sample

    # 修复：掩码区域用预测值，未遮挡区域保持原图
    x = x_prev * mask + x_0 * (1 - mask)

    # === RePaint 跳步逻辑 ===
    if (i > 0) and (i % jump_every == 0) and (jump_counter < max_jumps):
        jump_counter += 1
        back_i = max(0, i - jump_n)
        t_back = scheduler.timesteps[back_i]

        with torch.no_grad():
            noise_pred = pipe.unet(x, t_back).sample
            x_forward = scheduler.step(noise_pred, t_back, x).prev_sample

        with torch.no_grad():
            noise_pred = pipe.unet(x_forward, t).sample
            x = scheduler.step(noise_pred, t, x_forward).prev_sample

        x = x * mask + x_0 * (1 - mask)

    # 可视化保存
    if i % save_interval == 0 or i == scheduler.num_inference_steps - 1:
        vis = x.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        step_images.append(vis)

# ========== 5. 显示最终修复结果 ==========
recon = x.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
orig = x_0.squeeze(0).permute(1, 2, 0).cpu().numpy()

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(orig)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(masked_image)
plt.title("Masked")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(recon)
plt.title("Inpainted (DDIM + RePaint)")
plt.axis("off")
plt.show()

# ========== 6. 显示修复过程 ==========
fig, axs = plt.subplots(1, len(step_images), figsize=(3 * len(step_images), 3))
for idx, img in enumerate(step_images):
    axs[idx].imshow(img)
    axs[idx].set_title(f"Step {idx * save_interval}")
    axs[idx].axis("off")
plt.suptitle("DDIM + RePaint Inference Progress")
plt.show()
