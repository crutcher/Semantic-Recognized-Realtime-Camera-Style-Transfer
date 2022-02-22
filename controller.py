import argparse
import json
import math
import random
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import launchpad_py
import paho.mqtt.client as mqtt


def colorclamp(v) -> int:
    return max(min(int(v), 64), 0)


@dataclass(frozen=True)
class Color:
    r: int
    g: int
    b: int


@dataclass(frozen=True)
class Point:
    x: int
    y: int

    def write(
        self,
        device: launchpad_py.LaunchpadPro,
        color: Color,
        scale=1.0,
    ):
        device.LedCtrlXYByRGB(
            x=self.x,
            y=self.y,
            lstColor=[
                colorclamp(scale * color.r),
                colorclamp(scale * color.g),
                colorclamp(scale * color.b),
            ],
        )


@dataclass(frozen=True)
class PixelState:
    point: Point
    color: Color

    def write(
        self,
        device: launchpad_py.LaunchpadPro,
        scale=1.0,
    ):
        self.point.write(
            device=device,
            color=self.color,
            scale=scale,
        )


class CharkaController:
    client: mqtt.Client
    device: launchpad_py.LaunchpadPro

    scale: float = 1.0

    color_holds: Dict[Point, PixelState]

    def __init__(
        self,
        args: argparse.Namespace,
        client: mqtt.Client,
    ):
        self.client = client
        self.color_holds = {}
        
        if launchpad_py.LaunchpadMiniMk3().Check():
            self.device = launchpad_py.LaunchpadMiniMk3()
            assert self.device.Open(1, "minimk3"), "Could not find Launchapd controller"
        elif launchpad_py.LaunchpadLPX().Check(1):
            self.device = launchpad_py.LaunchpadLPX()
            assert self.device.Open(1), "Could not find Launchapd controller"
        else:
            raise AssertionError("Could not find Launchpad controller")

        self.device.Reset()
        self.device.LedCtrlBpm(40)

    def bl_to_ul(self, x: int, y: int) -> Tuple[int, int]:
        return (x, 8 - y)

    def send(self, method, **kwargs):
        msg = json.dumps(
            dict(
                method=method,
                kwargs=kwargs,
            )
        )
        print(msg)
        self.client.publish(
            "care/chakralens/action",
            msg,
        )

    def on_press(self, x, y):
        if x == 0 and y == 8:
            # bottom left corner
            self.send("reset", fg=1, bg=0)

        if y > 0 and x < 8:
            self.color_holds.clear()
            p = Point(x, y)
            self.color_holds[p] = PixelState(p, Color(64, 64, 64))
            for p in (
                *(Point(col, y) for col in range(8) if col != x),
                *(Point(x, row) for row in range(1, 9) if row != y),
            ):
                self.color_holds[p] = PixelState(p, Color(0, 40, 40))

            fg = 7 - y
            bg = x
            self.send("set_lens", fg=fg, bg=bg)

        if x == 0 and y == 0:
            self.send("more_weird")

        if x == 1 and y == 0:
            self.send("less_weird")

        if x == 2 and y == 0:
            self.send("mirror")

        if x == 3 and y == 0:
            self.send("flip_fg_bg")

        if x == 4 and y == 0:
            self.send("swap")

        if x == 5 and y == 0:
            self.send("random_lens")

        if x == 6 and y == 0:
            self.send("set_lens", bg=-2)

        if x == 7 and y == 0:
            self.send("isolate_fg")

        if x == 8 and y == 1:
            self.send("set_lens", fg=-2)

        if x == 8 and y == 2:
            self.send("set_lens", fg=-3)

        if x == 8 and y == 3:
            self.send("set_lens", bg=-3)

        if x == 8 and y == 4:
            self.send("undo_lens")

        if x == 8 and y == 5:
            self.send("redo_lens")

        if x == 8 and y == 6:
            self.send("freeze")
            self.send("random_lens")
            self.send("isolate_fg")

        if x == 8 and y == 7:
            self.send("rotate_lens")

        if x == 8 and y == 8:
            self.send("freeze")

    def on_release(self, x, y):
        pass

    def run(self):
        self.device.Reset()
        self.send("reset", fg=1, bg=0)
        self.send("clear_undo_history")

        fbs = [
            (fg, bg) for fg in range(8) for bg in range(8) if not (fg == 0 and bg == 0)
        ]

        self.scale = 0.0

        def hold_check_write(pixel: PixelState):
            if pixel.point in self.color_holds:
                return
            pixel.write(self.device, scale=self.scale)

        def put_color(x, y, color):
            hold_check_write(PixelState(Point(x, y), color))

        def put(x, y, r, g, b):
            hold_check_write(PixelState(Point(x, y), Color(r, g, b)))

        random_colors = [
            Color(64, 0, 0),
            Color(0, 64, 0),
            Color(0, 0, 64),
            Color(64, 64, 0),
            Color(0, 64, 64),
        ]

        print("doing a loop")
        while True:
            if button_state := self.device.ButtonStateXY():
                x, y, down = button_state
                if down:
                    self.on_press(x=x, y=y)
                else:
                    self.on_release(x=x, y=y)

            time.sleep(0.075)
            now = time.time()
            self.scale = 0.66 + math.sin(now) / 3

            for pixel in self.color_holds.values():
                pixel.write(device=self.device, scale=self.scale)

            if self.color_holds and random.randint(0, 40) == 0:
                point = random.choice(list(self.color_holds))
                del self.color_holds[point]

            for fg, bg in random.choices(fbs, k=1):
                x, y = self.bl_to_ul(bg, fg)
                pixel = PixelState(
                    Point(x, y),
                    Color(
                        random.randint(30, 64),
                        random.randint(30, 50),
                        random.randint(30, 64),
                    ),
                )
                if pixel not in self.color_holds:
                    pixel.write(self.device, self.scale)

                    # sometimes, we save them.
                    if random.randint(0, 20) == 0:
                        self.color_holds[pixel.point] = pixel

            for fg, bg in random.choices(fbs, k=10):
                if fg == 0 and bg == 0:
                    continue

                x, y = self.bl_to_ul(bg, fg)
                put(
                    x=x,
                    y=y,
                    r=fg * 8,
                    g=0,
                    b=bg * 8,
                )

            # reset
            put(x=0, y=8, r=0, g=10, b=10)

            # weirder
            put(x=0, y=0, r=45, g=0, b=45)
            # tamer
            put(x=1, y=0, r=32, g=32, b=32)

            # mirror
            put(x=2, y=0, r=0, g=0, b=64)

            # flip fg
            put(x=3, y=0, r=0, g=45, b=45)

            # swap lens
            self.device.LedCtrlPulseXYByCode(4, 0, 64)

            # random
            if random.randint(0, 5) == 0:
                put_color(5, 0, random.choice(random_colors))

            # bg void
            # x = 6, y = 0
            put(
                x=6,
                y=0,
                r=random.randint(0, 10),
                g=0,
                b=random.randint(0, 10),
            )

            # isolate
            put(x=7, y=0, r=0, g=64, b=0)

            # logo
            put(x=8, y=0, r=32, g=0, b=64)

            # fg void
            # x = 8, y = 1
            put(
                x=8,
                y=1,
                r=0,
                g=random.randint(0, 10),
                b=random.randint(0, 10),
            )

            # random fg
            put(x=8, y=2, r=64, g=0, b=0)

            # random bg
            put(x=8, y=3, r=0, g=0, b=64)

            # redo
            put(x=8, y=4, r=20, g=0, b=20)

            # undo
            put(x=8, y=5, r=00, g=20, b=20)

            # freeze and isolate
            put(x=8, y=6, r=64, g=0, b=64)

            # next patch
            put(x=8, y=7, r=0, g=64, b=64)

            # stop
            put(x=8, y=8, r=64, g=0, b=0)


def main(argv):
    parser = argparse.ArgumentParser()
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

    client = mqtt.Client()
    client.connect(args.mqtt_broker, args.mqtt_port, 60)
    mqtt_thread = threading.Thread(
        target=client.loop_forever,
        daemon=True,
    )
    mqtt_thread.start()

    CharkaController(args, client).run()


if __name__ == "__main__":
    main(sys.argv)
