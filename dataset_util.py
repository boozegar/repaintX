import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np


class CelebAHQInpaintingDataset(Dataset):
    def __init__(self, image_dir, mask_generator, image_size=256,
                 split="train"):
        self.image_dir = os.path.join(image_dir,
                                      split if split != "val" else "validation")  # Assuming structure like celeba_hq/train, celeba_hq/validation
        if not os.path.exists(self.image_dir):
            # Fallback for common CelebA-HQ data directory structures
            self.image_dir = image_dir  # Or a specific path where all images are
            print(
                f"Warning: Split directory {os.path.join(image_dir, split)} not found. Using {self.image_dir} and hoping for the best with file names.")

        self.image_files = [os.path.join(self.image_dir, f) for f in
                            os.listdir(self.image_dir) if
                            f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # Simplified split for example, real CelebA-HQ often has specific train/val/test splits by index
        if split == "train":
            self.image_files = self.image_files[:28000]
        elif split == "val":
            self.image_files = self.image_files[28000:29000]  # Example split
        else:  # test
            self.image_files = self.image_files[29000:30000]

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # Scales to [0, 1]
            transforms.Normalize([0.5], [0.5])  # Scales to [-1, 1]
        ])
        self.mask_generator = mask_generator
        self.image_size = image_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            img_path = self.image_files[idx]
            original_image = Image.open(img_path).convert("RGB")
            original_image_tensor = self.transform(original_image)

            # Generate mask (1 for masked region, 0 for known region)
            # Mask should be (1, H, W)
            mask = self.mask_generator(
                self.image_size)  # Returns a [H, W] numpy array
            mask_tensor = torch.from_numpy(mask).unsqueeze(
                0).float()  # (1, H, W)

            masked_image_tensor = original_image_tensor * (
                        1.0 - mask_tensor)  # Known regions

            return {
                "original_image": original_image_tensor,  # [-1, 1]
                "mask": mask_tensor,  # [0, 1], 1 is hole
                "masked_image_for_cond": masked_image_tensor
                # [-1, 1], known regions
            }
        except Exception as e:
            print(f"Error loading image {self.image_files[idx]}: {e}")
            # Return a dummy item or skip
            dummy_img = torch.zeros(3, self.image_size, self.image_size)
            dummy_mask = torch.ones(1, self.image_size,
                                    self.image_size)  # All masked
            return {
                "original_image": dummy_img,
                "mask": dummy_mask,
                "masked_image_for_cond": dummy_img * (1.0 - dummy_mask)
            }


# Example Mask Generator (Random Rectangular Masks)
def random_rectangular_mask_generator(image_size, min_rects=1, max_rects=4,
                                      min_size_ratio=0.05, max_size_ratio=0.3):
    mask = np.zeros((image_size, image_size), dtype=np.float32)
    num_rects = random.randint(min_rects, max_rects)
    for _ in range(num_rects):
        rect_h = int(
            random.uniform(min_size_ratio, max_size_ratio) * image_size)
        rect_w = int(
            random.uniform(min_size_ratio, max_size_ratio) * image_size)
        start_y = random.randint(0, image_size - rect_h)
        start_x = random.randint(0, image_size - rect_w)
        mask[start_y: start_y + rect_h, start_x: start_x + rect_w] = 1.0
    return mask


# --- Hyperparameters --- (Copied from previous response for context)
config = {
    "image_size": 256,
    "batch_size": 8,  # Adjust based on GPU memory
    "learning_rate": 1e-4,
    "num_train_timesteps": 1000,
    "beta_schedule": "cosine",
    "num_epochs": 100,  # Example
    "dataset_path": "~/SHB/celeba_hq/images",  # IMPORTANT: Update this path
    "output_dir": "./diffusion_inpainting_celeba_hq",
    "mixed_precision": "fp16",  # or "bf16" or "no"
    "gradient_accumulation_steps": 1,
    "save_image_epochs": 10,
    "save_model_epochs": 20,
}
# Update dataset path if CelebA-HQ images are not in a direct 'images' subfolder
# For example, if they are directly in 'celeba_hq_dataset/img_celeba_hq_256x256/'
# config["dataset_path"] = "path/to/celeba_hq_dataset/img_celeba_hq_256x256/"


# (Instantiate dataset and dataloader later in the training script)
