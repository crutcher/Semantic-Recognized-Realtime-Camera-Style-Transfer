import argparse
import sys
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn

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

    def load(self) -> None:
        print(f"Loading transform style on device: {self.device}")
        self.vgg = style_model.make_vgg()
        self.vgg.load_state_dict(torch.load(VGG_CHECKPOINT))
        self.vgg.cuda(self.device).eval()

        children = list(self.vgg.children())
        self.enc_1_to_4 = nn.Sequential(*children[:31])  # input -> relu4_1
        self.enc_5 = nn.Sequential(*children[31:44])  # relu4_1 -> relu5_1

        self.transform = style_model.Transform(in_planes=512)
        self.transform.load_state_dict(torch.load(TRANSFORMER_CHECKPOINT))
        self.transform.cuda(self.device).eval()

        self.decoder = style_model.make_decoder()
        self.decoder.load_state_dict(torch.load(DECODER_CHECKPOINT))
        self.decoder.cuda(self.device).eval()

    def encode_frame(self, source: EncodedFrameOrTensor) -> EncodedFrame:
        if isinstance(source, EncodedFrame):
            return source

        e4 = self.enc_1_to_4(source)
        e5 = self.enc_5(e4)

        return EncodedFrame(e4, e5)

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


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--style_devices",
        nargs="+",
        help="Style devices",
        default=["cuda"],
    )

    args = parser.parse_args(argv[1:])

    style_devices = sorted(set(args.style_devices))

    style_runners = []
    for d in style_devices:
        runner = StyleRunner(torch.device(d))
        runner.load()
        style_runners.append(runner)

    print(style_runners)


if __name__ == "__main__":
    main(sys.argv)
