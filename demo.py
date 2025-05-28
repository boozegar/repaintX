import os
import torch
from datasets import load_dataset
from diffusers import UNet2DConditionModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm

# 加载数据集
ds = load_dataset("bitmind/celeb-a-hq")

# 数据预处理
image_size = 256
preprocess = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


def transform(examples):
    images = [preprocess(image) for image in examples["image"]]
    return {"images": images}


ds = ds.with_transform(transform)

# 创建数据加载器
train_dataloader = DataLoader(ds["train"], batch_size=8, shuffle=True)

# 加载预训练模型组件
model_id = "facebook/dit-base-finetuned-cifar10"  # 使用一个基础DiT模型作为起点
unet = UNet2DConditionModel.from_pretrained(model_id)
noise_scheduler = DDPMScheduler.from_pretrained(model_id)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

# 优化器和学习率调度器
optimizer = torch.optim.AdamW(
    unet.parameters(),
    lr=1e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    eps=1e-8
)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=(len(train_dataloader) * 100),
)

# 初始化accelerator用于分布式训练
accelerator = Accelerator(
    mixed_precision="fp16",
    gradient_accumulation_steps=1,
    log_with="tensorboard",
    project_dir="./logs"
)

# 准备模型和优化器
unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    unet, text_encoder, optimizer, train_dataloader, lr_scheduler
)

# 创建EMA模型
ema_unet = EMAModel(unet.parameters(), decay=0.9999)

# 训练循环
num_epochs = 100
global_step = 0

for epoch in range(num_epochs):
    progress_bar = tqdm(total=len(train_dataloader),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch {epoch}")

    unet.train()
    for step, batch in enumerate(train_dataloader):
        clean_images = batch["images"]

        # 样本噪声
        noise = torch.randn_like(clean_images)

        # 随机时间步
        bsz = clean_images.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                  (bsz,), device=clean_images.device).long()

        # 添加噪声
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        # 前向传播
        with accelerator.accumulate(unet):
            # 文本编码
            encoder_hidden_states = text_encoder(batch["text_input_ids"])[0]

            # 预测噪声
            noise_pred = unet(noisy_images, timesteps,
                              encoder_hidden_states).sample

            # 计算损失
            loss = torch.nn.functional.mse_loss(noise_pred, noise,
                                                reduction="mean")

            # 反向传播
            accelerator.backward(loss)

            # 梯度裁剪
            accelerator.clip_grad_norm_(unet.parameters(), 1.0)

            # 优化器更新
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # 更新EMA模型
            if accelerator.sync_gradients:
                ema_unet.step(unet.parameters())

        progress_bar.update(1)
        global_step += 1

        logs = {"loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

    # 保存模型
    if accelerator.is_main_process:
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            pipeline = DDPMPipeline(
                unet=ema_unet.averaged_model,
                scheduler=noise_scheduler,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
            )
            pipeline.save_pretrained(f"dit-celebahq-epoch-{epoch + 1}")

    progress_bar.close()