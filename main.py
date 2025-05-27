from diffusers import DDPMPipeline
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ========== 1. 加载预训练 DDIMPipeline ==========
model_id = "google/ddpm-ema-celebahq-256"
# load model and scheduler
pipe = DDPMPipeline.from_pretrained(model_id, allow_pickle=False).to('cuda')  #
scheduler = pipe.scheduler
scheduler.set_timesteps(300)

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
mask = 1 - mask
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


x_image = x.squeeze().permute(1, 2, 0).clamp(0, 1).cpu().numpy()
plt.imshow(x_image)
plt.title("x_image")
plt.axis("off")
# 获取当前时间并格式化为字符串
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# 构建带时间戳的文件名
filename = f"x_image_{current_time}.png"

# 保存图片（dpi参数可调整图片清晰度）
plt.savefig(filename, dpi=300, bbox_inches='tight')