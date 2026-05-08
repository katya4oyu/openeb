# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Minimal PySide6 event viewer using OpenEB Python bindings.

This sample intentionally avoids metavision_sdk_ui and OpenGL. Events are read
with EventsIterator, rendered into a NumPy BGR frame, converted to QImage, and
shown through Qt's regular raster widgets.
"""

import argparse
import sys

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QSizePolicy

from metavision_core.event_io import EventsIterator
from metavision_sdk_core import BaseFrameGenerationAlgorithm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal PySide6 event viewer for RAW/HDF5 event files.",
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
    args = parser.parse_args()
    if args.delta_t <= 0:
        parser.error("--delta-t must be a positive integer")
    if args.max_duration is not None and args.max_duration <= 0:
        parser.error("--max-duration must be a positive integer when provided")
    return args


def bgr_frame_to_qimage(frame):
    """Return a detached QImage for a contiguous HxWx3 uint8 BGR/RGB frame."""
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected an HxWx3 uint8 frame")

    contiguous = np.ascontiguousarray(frame)
    height, width, _ = contiguous.shape
    bytes_per_line = contiguous.strides[0]

    if hasattr(QImage.Format, "Format_BGR888"):
        image_format = QImage.Format.Format_BGR888
    else:
        contiguous = contiguous[:, :, ::-1].copy()
        bytes_per_line = contiguous.strides[0]
        image_format = QImage.Format.Format_RGB888

    return QImage(contiguous.data, width, height, bytes_per_line, image_format).copy()


class EventViewer(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("OpenEB PySide6 Event Viewer")

        self.iterator = EventsIterator(
            input_path=args.input_event_file,
            mode="delta_t",
            delta_t=args.delta_t,
            max_duration=args.max_duration,
        )
        height, width = self.iterator.get_size()
        if width is None or height is None:
            raise RuntimeError("Could not determine sensor geometry from the event stream")

        self.events = iter(self.iterator)
        self.frame = np.empty((height, width, 3), dtype=np.uint8)
        self.paused = False

        self.image_label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(width, height)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCentralWidget(self.image_label)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.timer.start(max(1, args.delta_t // 1000))

        self.shortcuts = [
            QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self.toggle_pause),
            QShortcut(QKeySequence(Qt.Key.Key_P), self, activated=self.toggle_pause),
            QShortcut(QKeySequence(Qt.Key.Key_Q), self, activated=self.close),
            QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=self.close),
        ]

    def toggle_pause(self):
        self.paused = not self.paused

    def next_frame(self):
        if self.paused:
            return

        try:
            events = next(self.events)
        except StopIteration:
            self.timer.stop()
            return

        BaseFrameGenerationAlgorithm.generate_frame(events, self.frame)
        image = bgr_frame_to_qimage(self.frame)
        pixmap = QPixmap.fromImage(image)
        if self.image_label.size() != pixmap.size():
            pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        self.image_label.setPixmap(pixmap)

    def resizeEvent(self, event):
        pixmap = self.image_label.pixmap()
        if pixmap is not None:
            self.image_label.setPixmap(
                pixmap.scaled(
                    self.image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )
            )
        super().resizeEvent(event)


def main():
    args = parse_args()
    app = QApplication(sys.argv)
    viewer = EventViewer(args)
    viewer.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
