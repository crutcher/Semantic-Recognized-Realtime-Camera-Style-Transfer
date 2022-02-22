import argparse
import itertools
import queue
import random
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import json5
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.utils.mobile_optimizer
import torchvision.transforms.functional
from torch.nn import functional as F

import style_model
from base.base_inference import VideoInference
from models import UNet

torch.inference_mode(True)

# Path of pretrained weights of vgg encoder for style transfer
VGG_CHECKPOINT = "model_checkpoints/vgg_normalized.pth"
# Path of pretrained weights of decoder for style transfer
DECODER_CHECKPOINT = "model_checkpoints/decoder.pth"
# Path of pretrained weights of transformer for style transfer
TRANSFORMER_CHECKPOINT = "model_checkpoints/transformer.pth"
# Path of pretrained weights of UNet for segmentation
SEGMENTATION_NET_CHECKPOINT = "model_checkpoints/UNet_ResNet18.pth"


# Device
device = "cuda" if torch.cuda.is_available() else "cpu"


def central_square_crop(img):
    center = (img.shape[0] / 2, img.shape[1] / 2)
    h = w = min(img.shape[0], img.shape[1])
    x = center[1] - w / 2
    y = center[0] - h / 2
    crop_img = img[int(y) : int(y + h), int(x) : int(x + w), :]
    return crop_img


# Reverse the normalized image to normal distribution for displaying
def denormalize(img):
    MEAN = [0, 0, 0]
    STD = [1, 1, 1]
    MEAN = [-mean / std for mean, std in zip(MEAN, STD)]
    STD = [1 / std for std in STD]
    denormalizer = torchvision.transforms.Compose(
        [torchvision.transforms.Normalize(mean=MEAN, std=STD)]
    )
    return denormalizer(img)


# Convert torch tensor back to OpenCV image for displaying
def to_cv2_image(tensor):
    # Remove the batch_size dimension
    tensor = tensor.squeeze()

    # RGB 2 BGR
    tensor = tensor.flip(0)

    # is tis needed at all?
    # tensor = denormalize(tensor)

    img = tensor.cpu().numpy()

    # Transpose from [C, H, W] -> [H, W, C]
    return img.transpose(1, 2, 0)


class SegModel:
    segmentation_model: UNet
    segment_model_driver: VideoInference

    @torch.inference_mode()
    def __init__(
        self,
        *,
        segmentation_size: int,
        out_height: int,
        out_width: int,
    ):
        # Load segmentation net model
        print("Loading Segmentation Network")
        self.segmentation_model = UNet(backbone="resnet18", num_classes=2)
        self.segmentation_model.cuda().eval()
        trained_dict = torch.load(SEGMENTATION_NET_CHECKPOINT)["state_dict"]
        self.segmentation_model.load_state_dict(trained_dict, strict=False)

        # Create segmentation object
        self.segment_model_driver = VideoInference(
            model=self.segmentation_model,
            video_path=0,
            input_size=segmentation_size,
            height=out_height,
            width=out_width,
            use_cuda=torch.cuda.is_available(),
            draw_mode="matting",
        )

        self.segment_model_driver.mean_t = torch.tensor(
            self.segment_model_driver.mean,
            dtype=torch.float32,
            device=device,
        ).permute((2, 0, 1))
        self.segment_model_driver.std_t = torch.tensor(
            self.segment_model_driver.std,
            dtype=torch.float32,
            device=device,
        ).permute((2, 0, 1))
        print("Done Loading Segmentation Network\n")

    def segment_tensor(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        # image = image.to(torch.float32)
        image = torchvision.transforms.functional.resize(
            image,
            (
                self.segment_model_driver.input_size,
                self.segment_model_driver.input_size,
            ),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
        )
        image = (
            image - self.segment_model_driver.mean_t
        ) / self.segment_model_driver.std_t
        mask = self.segment_model_driver.model(image.unsqueeze(0))

        if False:
            mask = F.softmax(mask, dim=1)
        elif False:
            min = mask.min()
            max = mask.max()
            mask.subtract_(min)
            mask.divide_(max - min)
        elif False:
            torch.clamp_(mask, -1, 1)
            mask = F.softmax(mask, dim=1)
        elif True:
            mask = torch.copysign(torch.sqrt(torch.abs(mask)), mask)
            mask = F.softmax(mask, dim=1)
        elif False:
            mask = F.softmax(mask, dim=1)
            torch.sin_((torch.pi / 2) * mask)
        elif False:
            mask = torch.copysign(torch.log_(torch.abs(mask) + 1), mask)
            mask = F.softmax(mask, dim=1)
        elif True:
            signs = torch.signbit(mask)
            torch.abs_(mask)
            mask.add_(1.0)
            torch.log_(mask)
            mask[signs] *= -1
            mask = F.softmax(mask, dim=1)
        elif False:
            mask = F.softmax(mask, dim=1)
            mask.subtract_(0.24)
            mask.multiply_(torch.pi)
            mask.tan_()
        elif False:
            mask = F.softmax(mask, dim=1)
            mask.multiply_(2)
            mask.subtract_(1)
            mask.pow_(3)
            mask.add_(1)
            mask.multiply_(0.5)
        elif False:
            mask.pow_(3)
            mask = F.softmax(mask, dim=1)

        mask = mask[:, 1, ...].unsqueeze(0)

        # mask = F.gausian_blur(mask, 5, 0)

        mask = F.interpolate(
            # needs (1, 1, h, w)
            mask,
            size=(self.segment_model_driver.H, self.segment_model_driver.W),
            # mode="area",
            # align_corners=True,
        )
        return mask.squeeze()


class StyleModels:
    vgg: nn.Module
    transform: nn.Module
    decoder: nn.Module

    def __init__(self):
        # Load style transfer net model
        print("Loading Style Transfer Network")

        self.vgg = style_model.vgg
        self.vgg.load_state_dict(torch.load(VGG_CHECKPOINT))
        self.vgg.cuda().eval()
        children = list(self.vgg.children())
        self.enc_1_to_4 = nn.Sequential(*children[:31])  # input -> relu4_1
        self.enc_5 = nn.Sequential(*children[31:44])  # relu4_1 -> relu5_1

        self.transform = style_model.Transform(in_planes=512)
        self.transform.load_state_dict(torch.load(TRANSFORMER_CHECKPOINT))
        self.transform.cuda().eval()

        self.decoder = style_model.decoder
        self.decoder.load_state_dict(torch.load(DECODER_CHECKPOINT))
        self.decoder.cuda().eval()

        print("Done Loading Style Transfer Network\n")

    def run_enc_1_to_4(self, input: torch.Tensor) -> torch.Tensor:
        return self.enc_1_to_4(input.unsqueeze(0))

    def run_enc_5(self, e4: torch.Tensor) -> torch.Tensor:
        return self.enc_5(e4)


STYLE_MODELS = StyleModels()


@dataclass
class StyleEncodedFrame:
    e4: torch.Tensor
    e5: torch.Tensor


def encode_frame(content) -> StyleEncodedFrame:
    assert content.ndim == 3, content.shape
    e4 = STYLE_MODELS.run_enc_1_to_4(content)
    e5 = STYLE_MODELS.run_enc_5(e4)
    return StyleEncodedFrame(e4, e5)


@dataclass
class LazyInput:
    input: torch.Tensor
    _encoded: Optional[StyleEncodedFrame] = None

    @property
    def encoded(self) -> StyleEncodedFrame:
        if self._encoded is None:
            self._encoded = encode_frame(self.input)

        return self._encoded


def load_encoded_style(img: Union[str, np.ndarray]) -> StyleEncodedFrame:
    if isinstance(img, str):
        img = cv2.imread(img)

    img = central_square_crop(img)
    img = cv2.resize(img, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torchvision.transforms.functional.to_tensor(img)
    tensor = tensor.to(device)

    return encode_frame(tensor)


def apply_transform(
    content: Union[torch.Tensor, StyleEncodedFrame],
    style: Union[torch.Tensor, StyleEncodedFrame],
    *,
    pow: int = 1,
) -> torch.Tensor:
    if isinstance(style, torch.Tensor):
        style = encode_frame(style)

    for idx in range(pow):
        if isinstance(content, torch.Tensor):
            content = encode_frame(content)

        t = STYLE_MODELS.transform(
            content.e4,
            style.e4,
            content.e5,
            style.e5,
        )

        content = STYLE_MODELS.decoder(t).squeeze()

    return content


class ChakraLens:
    window_name = "💎🔥💎🌊⚡💎⚡💎⭐💎"

    camera_device: int

    capture_width: int
    capture_height: int

    render_width: int
    render_height: int

    display_width: int
    display_height: int

    styles: List[StyleEncodedFrame]

    RANDOM = -3

    PASSTHROUGH = -1
    NOTHING = -2

    fg: int = PASSTHROUGH
    bg: int = PASSTHROUGH

    lens_undo_stack: List[Tuple[int, int]]

    mirror: bool = False
    flip: bool = False
    freeze: bool = False
    isolate_fg: bool = False

    halt: bool = False

    seg_input: torch.Tensor
    seg_mask_raw: torch.Tensor

    background_delay: float
    segmentation_delay: float

    feedback: float

    config: Dict[str, Any]

    cam_queue: queue.Queue[torch.Tensor]
    display_queue: queue.Queue[torch.Tensor]

    def __init__(self, args: argparse.Namespace):
        self.camera_device = args.camera
        self.segmentation_delay = args.segmentation_delay
        self.background_delay = args.background_delay

        self.cam_queue = queue.Queue(maxsize=1)
        self.display_queue = queue.Queue(maxsize=2)

        self.lens_undo_stack = []

        try:
            with open(args.config) as f:
                self.config = json5.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load config file: {args.config}") from e

        # Camera resolution
        self.capture_width = args.capture_width
        self.capture_height = args.capture_height

        # Expected resolution of output video
        # The architecture of SANet requires that width and height of output video must be a multiple of 16
        self.render_width = int(16 * (args.render_width // 16))
        self.render_height = int(16 * (args.render_height // 16))

        # Display resolution
        self.display_width = args.display_width
        self.display_height = args.display_height

        print("Output Size:", (self.render_width, self.render_height))

        self.seg_model = SegModel(
            segmentation_size=args.segmentation_size,
            out_height=self.render_height,
            out_width=self.render_width,
        )

        self.styles = []
        for path in self.config["styles"]:
            self.styles.append(load_encoded_style(path))

        self.action_reset()

    def parse_action(self, msg: Dict[str, Any]) -> None:
        name = msg.get("method", None)
        kwargs = msg.get("kwargs", {})

        method_name = f"action_{name}"
        if name is None or not hasattr(self, method_name):
            return

        try:
            getattr(self, method_name)(**kwargs)

        except Exception:
            pass

    def action_rotate_lens(self, inc: int = 7) -> None:
        print("rotate_lens")
        self.styles[:] = self.styles[inc:] + self.styles[:inc]

    def action_reset(
        self,
        *,
        fg: int = PASSTHROUGH,
        bg: int = PASSTHROUGH,
    ) -> None:
        print("reset")
        self.action_set_lens(
            fg=fg,
            bg=bg,
        )
        self.isolate_fg = False
        self.freeze = False
        self.mirror = True
        self.flip = False
        self.feedback = 0.08

    def save_current_lens(self, *, redo: bool = False):
        if redo:
            idx = -1
        else:
            idx = 0

        current = (self.fg, self.bg)
        if self.lens_undo_stack and self.lens_undo_stack[idx] == current:
            # elide duplicates
            return

        while len(self.lens_undo_stack) > 1000:
            self.lens_undo_stack.pop(0)

        if redo:
            self.lens_undo_stack.insert(0, current)
        else:
            self.lens_undo_stack.append(current)

    def action_set_lens(self, *, fg: int = None, bg: int = None) -> None:
        self.save_current_lens()

        if fg is not None:
            if fg is self.RANDOM:
                fg = random.randint(0, len(self.styles) - 1)
            assert fg >= -2 and fg < len(self.styles), fg
            self.fg = fg

        if bg is not None:
            if bg is self.RANDOM:
                while bg is self.RANDOM or bg == self.fg:
                    bg = random.randint(-1, len(self.styles) - 1)
            assert bg >= -2 and bg < len(self.styles), bg
            self.bg = bg

        print(f"set lens: fg={fg}, bg={bg}")

    def action_clear_undo_history(self) -> None:
        print("clear_undo_history")
        self.lens_undo_stack.clear()

    def action_undo_lens(self) -> None:
        if self.lens_undo_stack:
            self.save_current_lens(redo=True)
            self.fg, self.bg = self.lens_undo_stack.pop()
        print(f"undo_lens: fg={self.fg}, bg={self.bg}")

    def action_redo_lens(self) -> None:
        if self.lens_undo_stack:
            self.save_current_lens()
            self.fg, self.bg = self.lens_undo_stack.pop(0)
        print(f"redo_lens: fg={self.fg}, bg={self.bg}")

    def action_flip_fg_bg(self) -> None:
        print("flip fg")
        self.flip = not self.flip

    def action_swap(self) -> None:
        print("swap")

        self.action_set_lens(fg=self.bg, bg=self.fg)

    def action_random_lens(self) -> None:
        print("random lens")
        self.action_set_lens(
            fg=self.RANDOM,
            bg=self.RANDOM,
        )

    def action_isolate_fg(self):
        print("isolate fg")
        self.isolate_fg = not self.isolate_fg
        if self.isolate_fg and self.bg == self.PASSTHROUGH:
            self.action_set_lens(bg=self.RANDOM)

    def action_freeze(self):
        print("freeze")
        self.freeze = not self.freeze

    def action_mirror(self) -> None:
        print("mirror")
        self.mirror = not self.mirror

    def action_more_weird(self) -> None:
        self.feedback = min(self.feedback + 0.01, 0.5)
        print("more_weird:", self.feedback)

    def action_less_weird(self) -> None:
        self.feedback = max(self.feedback - 0.01, 0.01)
        print("less_weird:", self.feedback)

    def render(self, style: int, source: LazyInput) -> torch.Tensor:
        if style == self.PASSTHROUGH:
            return source.input

        if style == self.NOTHING:
            return torch.empty_like(source.input)

        return apply_transform(
            source.encoded,
            style=self.styles[style],
        )

    @torch.inference_mode()
    def cam_thread_run(self):
        try:
            # Set webcam settings
            cam = cv2.VideoCapture(self.camera_device)
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
            cam.set(cv2.CAP_PROP_BUFFERSIZE, 4)
            cam.set(cv2.CAP_PROP_FPS, 30)

            while not self.halt:
                # Get webcam input
                read_success, cam_input = cam.read()
                if not read_success:
                    break

                input_tensor = (
                    torchvision.transforms.functional.to_tensor(cam_input).to(device)
                    # BGR 2 RGB
                    .flip(0)
                )

                input_tensor = torchvision.transforms.functional.resize(
                    input_tensor,
                    (self.render_height, self.render_width),
                    torchvision.transforms.InterpolationMode.BICUBIC,
                )

                # drop frames if the queue is full.
                try:
                    self.cam_queue.put(input_tensor, block=False)
                except queue.Full:
                    pass

            cam.release()

        except Exception as e:
            sys.exit(1)

    @torch.inference_mode()
    def segmentation_thread_run(self):
        last_seg_input = None

        try:
            while not self.halt:
                time.sleep(self.segmentation_delay)

                if self.seg_input is last_seg_input:
                    continue

                if self.seg_input is not None:
                    last_seg_input = self.seg_input

                    # don't segment when the displays are the same.
                    if self.isolate_fg or self.fg == self.NOTHING or self.fg != self.bg:
                        # Use segmentation to generate human_mask and bg_mask
                        # Shape of mask image is (H,W) and range is 0 ~ 255
                        seg_mask_raw = self.seg_model.segment_tensor(self.seg_input)

                        if self.seg_mask_raw is not None:
                            seg_mask_raw.multiply_(2)
                            seg_mask_raw.add_(self.seg_mask_raw)
                            seg_mask_raw.divide_(3)

                        self.seg_mask_raw

                        self.seg_mask_raw = seg_mask_raw

        except Exception as e:
            sys.exit(1)

    @torch.inference_mode()
    def style_thread_run(self):
        self.halt = False

        bg_tensor = None
        bg_tensor_ts = 0

        input_tensor = None
        content_tensor = None

        # Main loop
        try:
            for idx in itertools.count(start=0):
                if self.halt:
                    break

                if idx % 50:
                    # couldn't hurt?
                    torch.cuda.empty_cache()

                loop_start = time.time()

                cam_tensor = self.cam_queue.get()
                if input_tensor is None or not self.freeze:
                    input_tensor = cam_tensor

                self.seg_input = input_tensor

                source_tensor = input_tensor

                if self.feedback and content_tensor is not None:
                    source_tensor.multiply_(1.0 - self.feedback)
                    source_tensor.add_(content_tensor, alpha=self.feedback)

                lazy_input = LazyInput(source_tensor)

                content_tensor = self.render(self.fg, lazy_input)

                if self.flip:
                    content_tensor = content_tensor.flip(2)

                seg_mask = self.seg_mask_raw

                if seg_mask is not None and (
                    self.fg == self.NOTHING or self.fg != self.bg or self.isolate_fg
                ):
                    if self.flip:
                        seg_mask = seg_mask.flip(1)

                    if (
                        bg_tensor is None
                        or loop_start > bg_tensor_ts + self.background_delay
                    ):
                        if self.isolate_fg:
                            if bg_tensor is None:
                                bg_tensor = source_tensor

                            bg_tensor.multiply_(0.75)
                            bg_tensor.add_(
                                torch.rand_like(lazy_input.input),
                                alpha=0.25,
                            )

                            lazy_input = LazyInput(bg_tensor)

                        bg_tensor_ts = loop_start
                        bg_tensor = self.render(self.bg, lazy_input)

                    content_tensor.subtract_(bg_tensor)
                    content_tensor.multiply_(seg_mask)
                    content_tensor.add_(bg_tensor)

                torch.clamp_(content_tensor, 0.0, 1.0)

                self.display_queue.put(content_tensor)

        except Exception as e:
            sys.exit(1)

    def run(self):
        print("looping ...")
        cam_thread = threading.Thread(
            target=self.cam_thread_run,
            daemon=True,
        )
        cam_thread.start()

        self.seg_input = None
        self.seg_mask_raw = None
        seg_thread = threading.Thread(
            target=self.segmentation_thread_run,
            daemon=True,
        )
        seg_thread.start()

        style_thread = threading.Thread(
            target=self.style_thread_run,
            daemon=True,
        )
        style_thread.start()

        pygame.init()
        display = pygame.display.set_mode(
            size=(self.display_width, self.display_height),
            depth=32,
            flags=pygame.RESIZABLE,
        )
        pygame.display.set_caption("💎🔥🌊💨✨")

        loop_delay_times = []

        while not self.halt:
            loop_start = time.time()

            content_tensor = self.display_queue.get()

            # Take a horizontally mirrored view.
            if self.mirror:
                content_tensor = content_tensor.flip(2)

            content_tensor = torchvision.transforms.functional.resize(
                content_tensor,
                (display.get_height(), display.get_width()),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            )
            torch.clamp_(content_tensor, 0.0, 1.0)

            display_img = to_cv2_image(content_tensor)

            # pygame requires you build the target surface yourself.
            # display_img = cv2.resize(
            #     to_cv2_image(content_tensor),
            #     (display.get_width(), display.get_height()),
            #     cv2.INTER_CUBIC,
            # )

            # display_img = cv2.GaussianBlur(display_img, (5, 5), 0)
            # cv2.GaussianBlur(src=display_img, dst=display_img, ksize=(5, 5), sigmaX=0, sigmaY=0)
            # cv2.dilate(src=display_img, dst=display_img, kernel=(5, 5))

            # H, W, BGR, float32
            np.multiply(display_img, 255.0, out=display_img)
            display_img = display_img.transpose(1, 0, 2)[:, :, ::-1]
            # W, H, RGB, float32

            # ahoy, color magic here.
            surface = pygame.surfarray.make_surface(display_img)

            display.blit(surface, (0, 0))
            pygame.display.flip()

            events = pygame.event.get()
            key = -1
            for event in events:
                if event.type == pygame.KEYDOWN:
                    key = event.key

            loop_delay_times.append(time.time() - loop_start)
            loop_delay_times = loop_delay_times[-5:]

            fps = 1.0 / np.average(loop_delay_times)
            print(f"\rfps: {fps:.2f}", end="")

            if key >= ord("0") and key <= ord("8"):
                self.parse_action(
                    {
                        "method": "set_lens",
                        "kwargs": {"fg": int(chr(key)) - 2},
                    }
                )

            if key == ord("9"):
                self.parse_action({"method": "reset"})

            elif key == ord("i"):
                self.parse_action({"method": "isolate_fg"})

            elif key == ord("m"):
                self.parse_action({"method": "mirror"})

            elif key == ord("z"):
                self.parse_action({"method": "freeze"})

            elif key == ord("f"):
                self.parse_action({"method": "flip_fg_bg"})

            elif key == ord("s"):
                self.parse_action({"method": "swap"})

            elif key == 0x20:  # space
                self.parse_action({"method": "freeze"})
                self.parse_action({"method": "random_lens"})
                self.parse_action({"method": "isolate_fg_bg"})

            elif key == ord("r"):
                self.parse_action({"method": "random_lens"})

            elif key == ord("w"):
                self.parse_action({"method": "more_weird"})

            elif key == ord("n"):
                self.parse_action({"method": "less_weird"})

            elif key == ord("a"):
                pygame.display.toggle_fullscreen()

            elif key in [27, ord("q"), ord("e")]:
                print("escape")
                self.halt = True

        # drain the queues that may be blocking.
        try:
            self.cam_queue.get(block=True, timeout=0.5)
        except queue.Empty:
            pass
        try:
            self.display_queue.get(block=True, timeout=0.5)
        except queue.Empty:
            pass

        cam_thread.join()
        seg_thread.join()
        style_thread.join()

        cv2.destroyAllWindows()
        sys.exit(0)


def mqtt_thread_run(args: argparse.Namespace, chakra_lens: ChakraLens) -> None:
    print("starting mqtt client")
    import paho.mqtt.client as mqtt

    # The callback for when the client receives a CONNACK response from the server.
    def on_connect(client, userdata, flags, rc):
        print("Connected with result code " + str(rc))
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        client.subscribe("care/chakralens/action")

    # The callback for when a PUBLISH message is received from the server.
    def on_message(client, userdata, msg):
        try:
            chakra_lens.parse_action(json5.loads(msg.payload))
        except Exception as e:
            print(msg, e)

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(args.mqtt_broker, args.mqtt_port, 60)

    # Blocking call that processes network traffic, dispatches callbacks and
    # handles reconnecting.
    # Other loop*() functions are available that give a threaded interface and a
    # manual interface.
    client.loop_forever()


@torch.inference_mode()
def main(argv):
    # Argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        help="Path to config.",
        default="style_config.json",
    )
    parser.add_argument(
        "--camera",
        type=int,
        help="Camera device.",
        default=0,
    )
    parser.add_argument(
        "--segmentation_delay",
        type=float,
        help="Delay before regenerating the segmentation.",
        default=0.150,
    )
    parser.add_argument(
        "--background_delay",
        type=float,
        help="Delay before regenerating the background.",
        default=0.3,
    )
    parser.add_argument(
        "--capture_width",
        type=int,
        default=800,
        help="Camera capture width.",
    )
    parser.add_argument(
        "--capture_height",
        type=int,
        default=450,
        help="Camera capture height.",
    )
    parser.add_argument(
        "--display_width",
        type=int,
        default=1600,  # 1280,  # 1280,
        help="Display resize width",
    )
    parser.add_argument(
        "--display_height",
        type=int,
        default=900,  # 720,  # 720,
        help="Display resize height",
    )

    parser.add_argument(
        "--render_width",
        type=int,
        default=992,
        help="Size to render at (will be truncated to multiples of 16).",
    )
    parser.add_argument(
        "--render_height",
        type=int,
        default=558,
        help="Size to render at (will be truncated to multiples of 16).",
    )
    parser.add_argument(
        "--segmentation_size",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--mqtt_broker",
        type=str,
        default="localhost",
    )
    parser.add_argument(
        "--mqtt_port",
        type=int,
        default=1883,
    )
    args = parser.parse_args(argv[1:])

    chakra_lens = ChakraLens(args)

    if args.mqtt_broker is not None:
        mqtt_thread = threading.Thread(
            target=mqtt_thread_run,
            kwargs=dict(
                args=args,
                chakra_lens=chakra_lens,
            ),
            daemon=True,
        )
        mqtt_thread.start()

    chakra_lens.run()


if __name__ == "__main__":
    main(sys.argv)
