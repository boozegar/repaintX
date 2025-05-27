# !pip install diffusers
from diffusers import DiffusionPipeline
import PIL.Image
import numpy as np

model_id = "fusing/ddim-celeba-hq"

# load model and scheduler
ddpm = DiffusionPipeline.from_pretrained(model_id)

# run pipeline in inference (sample random noise and denoise)
image = ddpm(eta=0.0, num_inference_steps=50)

# process image to PIL
image_processed = image.cpu().permute(0, 2, 3, 1)
image_processed = (image_processed + 1.0) * 127.5
image_processed = image_processed.numpy().astype(np.uint8)
image_pil = PIL.Image.fromarray(image_processed[0])

# save image
image_pil.save("test.png")
