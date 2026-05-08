# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
EventToVideo flow inference sample.
"""

import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QSizePolicy

from metavision_core.event_io import AdaptiveRateEventsIterator, EventsIterator
from metavision_core_ml.event_to_video.lightning_model import EventToVideoLightningModel
from metavision_core_ml.preprocessing.event_to_tensor_torch import event_cd_to_torch, event_volume
from metavision_core_ml.utils.torch_ops import infer_device, normalize_tiles, viz_flow


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run an EventToVideo checkpoint on event slices and visualize its predicted optical flow.")
    parser.add_argument("path", type=str, help="Path to an event file readable by EventsIterator.")
    parser.add_argument("checkpoint", type=str, help="Path to an EventToVideo checkpoint.")
    parser.add_argument("--start_ts", type=int, default=0, help="Timestamp to start reading events from, in us.")
    parser.add_argument("--max_duration", type=int, default=-1,
                        help="Maximum duration to process after start_ts, in us. Negative means no limit.")
    parser.add_argument("--mode", type=str, default="mixed", choices=["n_events", "delta_t", "mixed", "adaptive"],
                        help="Event slicing mode.")
    parser.add_argument("--delta_t", type=int, default=30000, help="Slice duration in us for delta_t/mixed modes.")
    parser.add_argument("--n_events", type=int, default=30000, help="Event count per slice for n_events/mixed modes.")
    parser.add_argument("--thr_var", type=float, default=3e-5,
                        help="Variance threshold per event for adaptive slicing.")
    parser.add_argument("--height_width", nargs=2, default=None, type=int, metavar=("HEIGHT", "WIDTH"),
                        help="Resize network input and output to this resolution.")
    parser.add_argument("--video_path", type=str, default="", help="Optional path to an output video.")
    parser.add_argument("--no_window", action="store_true", help="Disable live display.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument("--device", type=str, default="", help="Torch device to use, for example cuda, mps or cpu.")
    parser.add_argument("--viz_input", action="store_true", help="Add a normalized event-volume panel.")
    parser.add_argument("--viz_gray", action="store_true", help="Add the reconstructed grayscale panel.")
    parser.add_argument("--no_flow", action="store_true", help="Do not show the predicted optical flow panel.")

    return parser.parse_args(argv)


def make_events_iterator(params):
    if params.mode == "adaptive":
        return AdaptiveRateEventsIterator(params.path, thr_var_per_event=params.thr_var)

    max_duration = None if params.max_duration < 0 else params.max_duration
    return EventsIterator(params.path, start_ts=params.start_ts, mode=params.mode, n_events=params.n_events,
                          delta_t=params.delta_t, max_duration=max_duration)


def events_to_network_input(events, height, width, out_height, out_width, nbins, device):
    events_th = event_cd_to_torch(events).to(device)
    start_times = torch.tensor([events["t"][0]], dtype=torch.float32, device=device)
    durations = torch.tensor([max(1, events["t"][-1] - events["t"][0])], dtype=torch.float32, device=device)
    tensor = event_volume(events_th, 1, height, width, start_times, durations, nbins, "bilinear")
    if (height, width) != (out_height, out_width):
        tensor = F.interpolate(tensor, size=(out_height, out_width), mode="bilinear", align_corners=True)
    return tensor.view(1, 1, nbins, out_height, out_width)


def gray_to_rgb(gray):
    gray = normalize_tiles(gray).squeeze().detach().cpu().numpy()
    gray = np.uint8(255 * gray)
    return np.repeat(gray[..., None], 3, axis=2)


def input_to_rgb(tensor):
    image = normalize_tiles(tensor.mean(dim=2), num_dims=3, num_stds=9).squeeze()
    image = np.uint8(255 * image.detach().cpu().numpy())
    return np.repeat(image[..., None], 3, axis=2)


def flow_to_rgb(flow):
    return viz_flow(flow).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()


def rgb_frame_to_qimage(frame):
    contiguous = np.ascontiguousarray(frame)
    height, width, _ = contiguous.shape
    return QImage(contiguous.data, width, height, contiguous.strides[0], QImage.Format.Format_RGB888).copy()


class FrameWindow(QMainWindow):
    def __init__(self, title):
        super().__init__()
        self.setWindowTitle(title)
        self.closed = False
        self.paused = False
        self.last_image = None
        self.image_label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCentralWidget(self.image_label)
        self.shortcuts = [
            QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self.toggle_pause),
            QShortcut(QKeySequence(Qt.Key.Key_P), self, activated=self.toggle_pause),
            QShortcut(QKeySequence(Qt.Key.Key_Q), self, activated=self.close),
            QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=self.close),
        ]

    def toggle_pause(self):
        self.paused = not self.paused

    def show_frame(self, image):
        self.last_image = image
        if self.image_label.minimumSize().isEmpty():
            self.image_label.setMinimumSize(image.width(), image.height())
        pixmap = QPixmap.fromImage(image)
        if self.image_label.size() != pixmap.size():
            pixmap = pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.FastTransformation)
        self.image_label.setPixmap(pixmap)

    def resizeEvent(self, event):
        if self.last_image is not None:
            pixmap = QPixmap.fromImage(self.last_image).scaled(self.image_label.size(),
                                                               Qt.AspectRatioMode.KeepAspectRatio,
                                                               Qt.TransformationMode.FastTransformation)
            self.image_label.setPixmap(pixmap)
        super().resizeEvent(event)

    def closeEvent(self, event):
        self.closed = True
        super().closeEvent(event)


class FrameSink:
    """PySide6 display plus optional video output."""

    def __init__(self, window_name, video_path):
        if not window_name and not video_path:
            raise ValueError("A window name or video path is required.")
        self.app = QApplication.instance()
        if window_name and self.app is None:
            self.app = QApplication(sys.argv)
        self.window = FrameWindow(window_name) if window_name else None
        self.video_path = video_path
        self.video_writer = None
        if self.window is not None:
            self.window.show()

    @property
    def closed(self):
        return bool(self.window is not None and self.window.closed)

    @property
    def paused(self):
        return bool(self.window is not None and self.window.paused)

    def __call__(self, image):
        if self.window is not None:
            self.window.show_frame(rgb_frame_to_qimage(image))
            self.app.processEvents()
        if self.video_path:
            if self.video_writer is None:
                import cv2

                height, width = image.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.video_writer = cv2.VideoWriter(self.video_path, fourcc, 25.0, (width, height))
            self.video_writer.write(image[:, :, ::-1])

    def close(self):
        if self.video_writer is not None:
            self.video_writer.release()
        if self.window is not None:
            self.window.close()


def run(params):
    if params.no_window and not params.video_path:
        raise ValueError("Use --video_path when --no_window is set.")
    if params.no_flow and not params.viz_gray and not params.viz_input:
        raise ValueError("Enable at least one output panel when --no_flow is set.")

    events_iterator = make_events_iterator(params)
    height, width = events_iterator.get_size()
    out_height, out_width = (height, width) if params.height_width is None else params.height_width

    device = infer_device(params.cpu, params.device or None)
    model = EventToVideoLightningModel.load_from_checkpoint(params.checkpoint, map_location=device)
    model.eval().to(device)
    nbins = model.hparams.event_volume_depth

    window_name = None if params.no_window else "metavision_flow_inference"
    frame_sink = FrameSink(window_name, params.video_path)

    print("Input size: {}x{}".format(height, width))
    print("Inference size: {}x{}".format(out_height, out_width))
    print("Event volume bins: {}".format(nbins))
    print("Device: {}".format(device))

    iterator = iter(events_iterator)
    with torch.no_grad():
        while True:
            if frame_sink.closed:
                break
            if frame_sink.paused:
                tensor = torch.zeros((1, 1, nbins, out_height, out_width), dtype=torch.float32, device=device)
            else:
                try:
                    events = next(iterator)
                except StopIteration:
                    break

                if len(events) == 0:
                    continue
                if events["t"][-1] < params.start_ts:
                    continue
                if params.mode == "adaptive" and params.max_duration >= 0:
                    if events["t"][-1] > params.start_ts + params.max_duration:
                        break
                tensor = events_to_network_input(events, height, width, out_height, out_width, nbins, device)

            state = model.model(tensor)
            panels = []
            if params.viz_input:
                panels.append(input_to_rgb(tensor))
            if not params.no_flow:
                panels.append(flow_to_rgb(model.model.predict_flow(state)))
            if params.viz_gray:
                panels.append(gray_to_rgb(model.model.predict_gray(state)))

            frame = np.concatenate(panels, axis=1) if len(panels) > 1 else panels[0]
            frame_sink(frame)

    frame_sink.close()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
