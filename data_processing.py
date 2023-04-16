import os
from PIL import Image
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image


def load_image(image_path, x32=False):
    img = Image.open(image_path).convert("RGB")

    if x32:
        def to_32s(x):
            return 256 if x < 256 else x - x % 32

        w, h = img.size
        img = img.resize((to_32s(w), to_32s(h)))

    return img

# def load_image(image_path, x32=False):
#     with Image.open(image_path) as img:
#         img = img.convert("RGB")
#
#         if x32:
#             def to_32s(x):
#                 return 256 if x < 256 else x - x % 32
#
#             w, h = img.size
#             img = img.resize((to_32s(w), to_32s(h)))
#
#         return img


def save_image(tensor, save_path):
    tensor = tensor.squeeze(0).clip(-1, 1) * 0.5 + 0.5
    img = to_pil_image(tensor)
    img.save(save_path)


def process_images(args, model):
    device = args.device
    model.to(device).eval()

    os.makedirs(args.output_dir, exist_ok=True)

    for image_name in sorted(os.listdir(args.input_dir)):
        if os.path.splitext(image_name)[-1].lower() not in [".jpg", ".png", ".bmp", ".tiff"]:
            continue

        image = load_image(os.path.join(args.input_dir, image_name), args.x32)

        with torch.no_grad():
            image = to_tensor(image).unsqueeze(0) * 2 - 1
            out = model(image.to(device), args.upsample_align).cpu()

        save_image(out, os.path.join(args.output_dir, image_name))
        print(f"Image saved: {image_name}")

