import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import make_image_grid  # For logging images
from accelerate import Accelerator  # For easy multi-GPU, mixed precision
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os
from torchvision.utils import save_image

from dataset_util import CelebAHQInpaintingDataset, \
    random_rectangular_mask_generator, config
from diffusion_atn import model


def train_inpainting_diffusion(config, model):
    accelerator = Accelerator(
        mixed_precision=config["mixed_precision"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        log_with="tensorboard",  # or "wandb"
        project_dir=os.path.join(config["output_dir"], "logs")
    )

    # Data
    train_dataset = CelebAHQInpaintingDataset(
        image_dir=config["dataset_path"],
        mask_generator=random_rectangular_mask_generator,  # Or your custom one
        image_size=config["image_size"],
        split="train"
    )
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config["batch_size"], shuffle=True,
                                  num_workers=4)

    # Noise Scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_train_timesteps"],
        beta_schedule=config["beta_schedule"]
    )

    # Optimizer and LR Scheduler
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config["learning_rate"])
    lr_scheduler = get_scheduler(
        "cosine",  # or "linear", "constant", etc.
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_dataloader) * config["num_epochs"]) //
                           config["gradient_accumulation_steps"],
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Parameter count after preparation (only on main process)
    if accelerator.is_main_process:
        total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Trainable Parameters: {total_params / 1e6:.2f}M")
        # For the example UNet config, this might be around 70-90M depending on exact head_dim choices.
        # E.g., the config with explicit AttnDown/UpBlock types results in ~87M params.

    global_step = 0
    for epoch in range(config["num_epochs"]):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader),
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description(
            f"Epoch {epoch + 1}/{config['num_epochs']}")

        for step, batch in enumerate(train_dataloader):
            original_images = batch["original_image"]  # [-1, 1]
            masks = batch["mask"]  # [0, 1], 1 is hole
            known_masked_images = batch[
                "masked_image_for_cond"]  # [-1, 1] (original_image * (1-mask))

            # Sample noise to add to the images
            noise = torch.randn_like(original_images)
            bs = original_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0,
                                      noise_scheduler.config.num_train_timesteps,
                                      (bs,),
                                      device=original_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(original_images, noise,
                                                     timesteps)

            # Prepare UNet input: concat(noisy_image, mask, original_masked_image)
            # noisy_image: What the model tries to denoise
            # mask: Tells the model WHICH regions are masked (holes)
            # known_masked_images: Provides context of known (unmasked) regions from original
            model_input = torch.cat([noisy_images, masks, known_masked_images],
                                    dim=1)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(model_input, timesteps).sample
                loss = F.mse_loss(noise_pred, noise)  # Predict the added noise

                accelerator.backward(loss)
                if accelerator.sync_gradients:  # Only clip when gradients are synced
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        progress_bar.close()

        # --- Evaluation and Saving (on main process) ---
        if accelerator.is_main_process:
            if (epoch + 1) % config["save_image_epochs"] == 0 or epoch == \
                    config["num_epochs"] - 1:
                evaluate_and_save_images(model, noise_scheduler, batch, config,
                                         epoch, accelerator.device)

            if (epoch + 1) % config["save_model_epochs"] == 0 or epoch == \
                    config["num_epochs"] - 1:
                pipeline_dir = os.path.join(config["output_dir"],
                                            f"epoch_{epoch + 1}")
                os.makedirs(pipeline_dir, exist_ok=True)
                # Save the UNet model (unwrap if using DDP)
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(pipeline_dir)
                print(f"Saved model to {pipeline_dir}")

    accelerator.wait_for_everyone()
    accelerator.end_training()
    print("Training finished.")


@torch.no_grad()
def evaluate_and_save_images(model, noise_scheduler, sample_batch, config,
                             epoch, device):
    model.eval()  # Set model to evaluation mode

    original_images_eval = sample_batch["original_image"][:4].to(
        device)  # Take a few samples
    masks_eval = sample_batch["mask"][:4].to(device)
    known_masked_images_eval = sample_batch["masked_image_for_cond"][:4].to(
        device)

    # Start with random noise in the masked region, and original image in unmasked region
    # For a pure generation in masked area, start with full noise in masked area
    noisy_sample = torch.randn_like(original_images_eval)
    # Initially, latents are pure noise in masked area, and ground truth in known area.
    # However, the diffusion process starts from T and denoises the *entire* image,
    # but we enforce data consistency.
    # Let's start with full noise and denoise the whole thing, then apply consistency.
    latents = torch.randn_like(original_images_eval)

    num_inference_steps = 50  # or 100, or config.num_train_timesteps for DDPM
    noise_scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(noise_scheduler.timesteps, desc="Sampling"):
        # 1. Prepare model input
        model_input = torch.cat([latents, masks_eval, known_masked_images_eval],
                                dim=1)

        # 2. Predict noise residual
        noise_pred = model(model_input, t).sample

        # 3. Compute previous noisy sample x_t -> x_{t-1}
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

        # 4. Data consistency: Inpaint known regions (VERY IMPORTANT for inpainting)
        # Ensure the unmasked parts of latents remain consistent with the original image's unmasked parts
        # At each step, the known regions should be from the *original image* (or its noised version at step t if preferred)
        # Simplest: directly use original known parts
        latents = latents * masks_eval + known_masked_images_eval * (
                1.0 - masks_eval)
        # More sophisticated (RePaint):
        # known_latents_at_t = noise_scheduler.add_noise(original_images_eval, torch.randn_like(original_images_eval), torch.tensor([t]*latents.shape[0], device=device))
        # latents = latents * masks_eval + known_latents_at_t * (1.0 - masks_eval)

    # Scale back to [0, 1] for saving
    inpainted_images = (latents / 2 + 0.5).clamp(0, 1)
    original_display = (original_images_eval / 2 + 0.5).clamp(0, 1)
    masked_display = (known_masked_images_eval / 2 + 0.5).clamp(0,
                                                                1)  # This is original_image * (1-mask)

    # Create a grid
    # Order: Original | Masked Input | Inpainted
    grid = make_image_grid([original_display, masked_display, inpainted_images],
                           rows=original_display.shape[0], cols=3)

    # Save the image
    save_dir = os.path.join(config["output_dir"], "samples")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"epoch_{epoch + 1}_sample.png")
    save_image(grid, save_path)
    print(f"Saved sample image to {save_path}")
    model.train()  # Set model back to train mode


# --- Main Execution ---
if __name__ == '__main__':
    # IMPORTANT: Update config["dataset_path"] to your CelebA-HQ image directory
    # e.g., "/path/to/celeba_hq/img_celeba_hq/" or if you have split folders
    # "/path/to/celeba_hq/" (and Dataset class handles train/validation subfolders)
    # config["dataset_path"] = "/mnt/data/celeba_hq_256x256_images/" # Example path
    if not os.path.exists(config["dataset_path"]) or len(
            os.listdir(config["dataset_path"])) == 0:
        print(
            f"ERROR: Dataset path {config['dataset_path']} is invalid or empty.")
        print(
            "Please update config['dataset_path'] to point to your CelebA-HQ images.")
        exit()

    os.makedirs(config["output_dir"], exist_ok=True)
    train_inpainting_diffusion(config, model)
