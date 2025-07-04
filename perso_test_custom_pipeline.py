import sys

sys.path.append("src")

import torch

from diffusers import FluxKontextPipeline
from diffusers.utils import load_image


pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16,
    custom_pipeline="/workspaces/diffusers/examples/community/pipeline_flux_kontext_multiple_images.py",
)
pipe.to("cuda")

pikachu_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
).convert("RGB")
cat_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
).convert("RGB")


prompts = [
    "Pikachu and the cat are sitting together at a pizzeria table, enjoying a delicious pizza.",
]
images = pipe(
    multiple_images=[(pikachu_image, cat_image)],
    prompt=prompts,
    guidance_scale=2.5,
    generator=torch.Generator().manual_seed(42),
).images
images[0].save("pizzeria.png")
