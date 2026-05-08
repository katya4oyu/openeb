# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Minimal OpenCV event viewer using OpenEB Python bindings.

This sample avoids metavision_sdk_ui and OpenGL. It reads RAW/HDF5 event files
with EventsIterator, renders slices into a NumPy BGR frame, and displays them
with cv2.imshow.
"""

import argparse
import time

import cv2
import numpy as np

from metavision_core.event_io import EventsIterator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal OpenCV event viewer for RAW/HDF5 event files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-event-file",
        required=True,
        help="Path to an input event file supported by EventsIterator, such as RAW or HDF5.",
    )
    parser.add_argument(
        "--delta-t",
        type=int,
        default=10000,
        help="Event slice duration in microseconds.",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=None,
        help="Maximum playback duration in microseconds. Omit to read until the file ends.",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Sleep between slices to approximate the event stream timing.",
    )
    args = parser.parse_args()
    if args.delta_t <= 0:
        parser.error("--delta-t must be a positive integer")
    if args.max_duration is not None and args.max_duration <= 0:
        parser.error("--max-duration must be a positive integer when provided")
    return args


def render_events(events, frame):
    frame.fill(0)
    if events.size == 0:
        return

    height, width = frame.shape[:2]
    xs = events["x"].astype(np.int64, copy=False)
    ys = events["y"].astype(np.int64, copy=False)
    ps = events["p"].astype(bool, copy=False)

    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs = xs[valid]
    ys = ys[valid]
    ps = ps[valid]

    frame[ys[~ps], xs[~ps]] = (255, 0, 0)
    frame[ys[ps], xs[ps]] = (0, 255, 0)


def main():
    args = parse_args()
    iterator = EventsIterator(
        input_path=args.input_event_file,
        mode="delta_t",
        delta_t=args.delta_t,
        max_duration=args.max_duration,
    )

    height, width = iterator.get_size()
    if width is None or height is None:
        raise RuntimeError("Could not determine sensor geometry from the event stream")

    frame = np.empty((height, width, 3), dtype=np.uint8)
    window_name = "OpenEB OpenCV Event Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    sleep_s = max(0.0, args.delta_t / 1_000_000.0)
    for events in iterator:
        render_events(events, frame)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if args.realtime:
            time.sleep(sleep_s)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
