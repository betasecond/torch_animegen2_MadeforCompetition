import argparse
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from model import Generator
from data_processing import load_image
import os


def test(args):
    device = args.device

    net = Generator()
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    net.to(device).eval()
    print(f"model loaded: {args.checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.input_image is not None:
        image_paths = [args.input_image]
    else:
        image_paths = [
            os.path.join(args.input_dir, image_name)
            for image_name in sorted(os.listdir(args.input_dir))
            if os.path.splitext(image_name)[-1].lower() in [".jpg", ".png", ".bmp", ".tiff"]
        ]

    for image_path in image_paths:
        image = load_image(image_path, args.x32)

        with torch.no_grad():
            image = to_tensor(image).unsqueeze(0) * 2 - 1
            out = net(image.to(device), args.upsample_align).cpu()
            out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
            out = to_pil_image(out)

        out.save(os.path.join(args.output_dir, os.path.basename(image_path)))
        print(f"image saved: {os.path.basename(image_path)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./weights/celeba_distill.pt',  # 更改为默认权重文件的路径
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='./samples/inputs',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./samples/results',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
    )
    parser.add_argument(
        '--upsample_align',
        type=bool,
        default=False,
        help="Align corners in decoder upsampling layers"
    )
    parser.add_argument(
        '--x32',
        action="store_true",
        help="Resize images to multiple of 32"
    )
    args = parser.parse_args()

    test(args)
