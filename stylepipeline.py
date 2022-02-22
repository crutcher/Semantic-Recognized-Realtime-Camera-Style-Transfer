import argparse
import sys
from dataclasses import dataclass
from typing import Union

import cv2
import json5
import numpy as np
import torch
import torch.nn as nn
import torchvision

import style_model

# Path of pretrained weights of vgg encoder for style transfer
VGG_CHECKPOINT = "model_checkpoints/vgg_normalized.pth"
# Path of pretrained weights of decoder for style transfer
DECODER_CHECKPOINT = "model_checkpoints/decoder.pth"
# Path of pretrained weights of transformer for style transfer
TRANSFORMER_CHECKPOINT = "model_checkpoints/transformer.pth"


class ModelRunner:
    device: torch.device

    def __init__(self, device: torch.device):
        self.device = device

    def load(self) -> None:
        raise NotImplemented


@dataclass
class EncodedFrame:
    e4: torch.Tensor
    e5: torch.Tensor


EncodedFrameOrTensor = Union[EncodedFrame, torch.Tensor]


class StyleRunner(ModelRunner):
    vgg: nn.Module
    transform: nn.Module
    decoder: nn.Module

    @torch.inference_mode()
    def load(self) -> None:
        print(f"Loading transform style on device: {self.device}")
        self.vgg = style_model.make_vgg()
        self.vgg.load_state_dict(torch.load(VGG_CHECKPOINT))
        self.vgg.cuda(self.device).eval()

        children = list(self.vgg.children())
        self.enc_1_to_4 = nn.Sequential(*children[:31])  # input -> relu4_1
        self.enc_1_to_4.cuda(self.device).eval()
        self.enc_5 = nn.Sequential(*children[31:44])  # relu4_1 -> relu5_1
        self.enc_5.cuda(self.device).eval()

        self.transform = style_model.Transform(in_planes=512)
        self.transform.load_state_dict(torch.load(TRANSFORMER_CHECKPOINT))
        self.transform.cuda(self.device).eval()

        self.decoder = style_model.make_decoder()
        self.decoder.load_state_dict(torch.load(DECODER_CHECKPOINT))
        self.decoder.cuda(self.device).eval()

    @torch.inference_mode()
    def encode_frame(self, source: EncodedFrameOrTensor) -> EncodedFrame:
        if isinstance(source, EncodedFrame):
            if source.e4.device == self.device:
                return source
            else:
                return EncodedFrame(
                    e4=torch.as_tensor(source.e4, device=self.device),
                    e5=torch.as_tensor(source.e5, device=self.device),
                )

        source = torch.as_tensor(source, device=self.device)

        e4 = self.enc_1_to_4(source.unsqueeze(0))
        e5 = self.enc_5(e4)

        return EncodedFrame(e4, e5)

    @torch.inference_mode()
    def apply(
        self,
        source: EncodedFrameOrTensor,
        style: EncodedFrameOrTensor,
    ) -> torch.Tensor:
        source = self.encode_frame(source)
        style = self.encode_frame(style)

        t = self.transform(
            source.e4,
            style.e4,
            source.e5,
            style.e5,
        )

        return self.decoder(t).squeeze()


def central_square_crop(img: np.ndarray) -> np.ndarray:
    center = (img.shape[0] / 2, img.shape[1] / 2)
    h = w = min(img.shape[0], img.shape[1])
    x = center[1] - w / 2
    y = center[0] - h / 2
    crop_img = img[int(y) : int(y + h), int(x) : int(x + w), :]
    return crop_img


def load_image_for_style(path: str) -> torch.Tensor:
    img = cv2.imread(path)
    img = central_square_crop(img)
    img = cv2.resize(img, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torchvision.transforms.functional.to_tensor(img)


@torch.inference_mode()
def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--style_devices",
        nargs="+",
        help="Style devices",
        default=["cuda"],
    )
    parser.add_argument(
        "--style_config",
        type=str,
        help="Path to config.",
        default="style_config.json",
    )

    args = parser.parse_args(argv[1:])

    style_devices = {
        name: torch.device(name) for name in sorted(set(args.style_devices))
    }

    style_runners = {}
    for name, device in style_devices.items():
        runner = StyleRunner(device)
        runner.load()
        style_runners[name] = runner

    try:
        with open(args.style_config) as f:
            style_config = json5.load(f)
    except Exception as e:
        raise ValueError(
            f"Failed to load style_config file: {args.style_config}"
        ) from e

    print("Loading cached styles")
    style_device_map = {}
    for style_path in style_config["styles"]:
        style_tensor = load_image_for_style(style_path)
        style_device_map[style_path] = {
            name: style_runners[name].encode_frame(style_tensor)
            for name, device in style_devices.items()
        }
    print(style_runners)


if __name__ == "__main__":
    main(sys.argv)
