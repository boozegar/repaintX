from diffusers import UNet2DModel

# 假设图像是 256x256, RGB (3 通道)
image_size = 256
in_image_channels = 3  # 原始图像通道
mask_channels = 1      # 掩码通道
# 输入给U-Net的通道数: noisy_image (3) + mask (1) + original_masked_image (3) = 7
# 或者，更常见的做法是: noisy_masked_image (3) + mask (1) = 4
# 我们这里采用论文中常见的做法，给U-Net输入 noisy_image, mask, original_masked_image
# 这样U-Net可以同时看到噪声模式、掩码区域以及已知区域的原始信息
unet_in_channels = in_image_channels + mask_channels + in_image_channels # 3+1+3=7


model = UNet2DModel(
    sample_size=image_size,
    in_channels=unet_in_channels,
    out_channels=in_image_channels,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 512, 512), # 减少一层以匹配attention_head_dim长度
    # 通常在特征图较小（例如16x16, 8x8）时加入注意力，因为计算成本较低且全局信息更重要
    # 假设 image_size=256, block_out_channels 对应的特征图大小依次为:
    # 256 (input) -> 128 (after first block type) -> 64 -> 32 -> 16 -> 8
    # 所以，如果 block_out_channels=(128, 128, 256, 512, 512)
    # feature map sizes might be: 128, 64, 32, 16, 8 (approx)
    # Let's add attention at resolutions where feature maps are 32x32, 16x16, 8x8
    attention_head_dim = (None, 8, 16, 16), # (Corresponds to layers with output channels 128(no attn), 256(attn), 512(attn), 512(attn))
                                           # Needs careful matching with block_out_channels and down_block_types/up_block_types if used manually
    # A simpler way to ensure attention is used is to use "AttnDownBlock2D" and "AttnUpBlock2D"
    # as shown in the commented out section.
    # For this example, let's re-enable the explicit block types for clarity:
    down_block_types=(
        "DownBlock2D",      # 256 -> 128
        "DownBlock2D",      # 128 -> 64
        "AttnDownBlock2D",  # 64 -> 32 (Add Attention)
        "AttnDownBlock2D",  # 32 -> 16 (Add Attention)
        "DownBlock2D",      # 16 -> 8
    ),
    up_block_types=(
        "UpBlock2D",        # 8 -> 16
        "AttnUpBlock2D",    # 16 -> 32 (Add Attention)
        "AttnUpBlock2D",    # 32 -> 64 (Add Attention)
        "UpBlock2D",        # 64 -> 128
        "UpBlock2D",        # 128 -> 256
    ),
)

# 打印模型参数量 (粗略，需要实例化后精确计算)
# print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
# (实际参数量取决于最终配置，上述配置大约在 30M-100M 范围，具体取决于block_out_channels的深度和宽度)