@torch.no_grad()
def sample_dit(model, diffusion, batch_size=4, device="cuda"):
    model.eval()
    # 从纯噪声开始
    x = torch.randn(batch_size, 3, 256, 256, device=device)

    # 逐步去噪
    for t in reversed(range(0, diffusion.timesteps)):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        x = diffusion.p_sample(model, x, t_batch)

    # 反归一化
    x = (x + 1.0) / 2.0
    x = torch.clamp(x, 0.0, 1.0)
    return x


def save_samples(images, path="samples.png"):
    from torchvision.utils import save_image
    save_image(images, path, nrow=2)


# 生成样本
samples = sample_dit(model, diffusion, batch_size=4, device=device)
save_samples(samples, "celebahq_samples.png")