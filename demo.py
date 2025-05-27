from diffusers import DDIMPipeline
import torch

model_id = "google/ddpm-ema-celebahq-256"

# 加载模型
ddpm = DDIMPipeline.from_pretrained(model_id).to('cuda')

# 设置采样步数为 1000
image = ddpm(num_inference_steps=500).images[0]

# 保存图像
image.save("ddpm_generated_image.png")
