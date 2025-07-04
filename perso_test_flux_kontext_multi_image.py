import sys

sys.path.append("src")

import torch
from PIL import Image

from diffusers import FluxKontextPipeline
from diffusers.utils import load_image


pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")

image1 = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
).convert("RGB").resize((1024, 1024), resample=Image.LANCZOS)
image2 = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
).convert("RGB").resize((1024, 1024), resample=Image.LANCZOS)
image3 = load_image(
    "https://www.pokemon.com/static-assets/content-assets/cms2/img/pokedex/full/151.png"
).convert("RGB").resize((1024, 1024), resample=Image.LANCZOS)


# prompts = [
#     "Change the visual style of the image of the cat to match the style of the Pikachu image. Keep the image composition the same.",
# ]
# images = pipe(
#     multiple_images=[(image1, image2)],
#     prompt=prompts,
#     guidance_scale=2.5,
#     generator=torch.Generator().manual_seed(42),
# ).images
# images[0].save("output_style_1.png")


prompts = [
    "Apply the Pikachu graphical effect on the cat",
]
images = pipe(
    multiple_images=[(image2, image1)],
    prompt=prompts,
    guidance_scale=2.5,
    generator=torch.Generator().manual_seed(42),
).images
images[0].save("output_style_2.png")




# prompts = [
#     "Put both animal in a cozy room",
# ]
# images = pipe(
#     multiple_images=[(image1, image2)],
#     prompt=prompts,
#     guidance_scale=2.5,
#     generator=torch.Generator().manual_seed(42),
# ).images
# images[0].save("output_0.png")


# prompts = [
#     "Put all animals in a cozy room",
# ]
# images = pipe(
#     multiple_images=[(image1, image2, image3)],
#     prompt=prompts,
#     guidance_scale=2.5,
#     generator=torch.Generator().manual_seed(42),
# ).images
# images[0].save("output_1.png")




# # join the two images
# image12 = Image.new("RGB", (image1.width + image2.width, image1.height))
# image12.paste(image1, (0, 0))
# image12.paste(image2, (image1.width, 0))
# print(image12.size)

# prompt = "Pikachu and a cat sitting together in a cozy room"
# image = pipe(
#     #  image=image1,
#     image=[[image1, image2]],
#      prompt=prompt,
#      guidance_scale=2.5,
#      generator=torch.Generator().manual_seed(42),
#  ).images[0]
# image.save("output.png")
