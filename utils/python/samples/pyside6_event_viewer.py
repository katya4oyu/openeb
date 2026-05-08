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
from RAW/HDF5 files or a live camera with EventsIterator, rendered into a NumPy
BGR frame, converted to QImage, and shown through Qt's regular raster widgets.
"""

import argparse
import sys
import time

import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QSizePolicy

from metavision_core.event_io import EventsIterator
from metavision_sdk_core import BaseFrameGenerationAlgorithm


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Minimal PySide6 event viewer for RAW/HDF5 event files or the first "
            "available live camera."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-event-file",
        default="",
        help=(
            "Path to an input event file supported by EventsIterator, such as RAW or HDF5. "
            "Omit to open the first available live camera."
        ),
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
        help="Maximum stream duration in microseconds. Omit to read until the file ends or the viewer is closed.",
    )
    parser.add_argument(
        "--display-fps",
        type=float,
        default=30.0,
        help="Maximum display refresh rate. Incoming slices above this rate are dropped to keep latency low.",
    )
    args = parser.parse_args()
    if args.delta_t <= 0:
        parser.error("--delta-t must be a positive integer")
    if args.max_duration is not None and args.max_duration <= 0:
        parser.error("--max-duration must be a positive integer when provided")
    if args.display_fps <= 0:
        parser.error("--display-fps must be a positive number")
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


class EventReaderWorker(QObject):
    frame_ready = Signal(QImage)
    error = Signal(str)
    finished = Signal()

    def __init__(self, iterator, height, width, display_fps):
        super().__init__()
        self.iterator = iterator
        self.frame = np.empty((height, width, 3), dtype=np.uint8)
        self.display_period_s = 1.0 / display_fps
        self.paused = False
        self.running = True

    @Slot()
    def run(self):
        events_iterator = iter(self.iterator)
        next_display_time = 0.0

        try:
            while self.running:
                if self.paused:
                    QThread.msleep(5)
                    next_display_time = time.monotonic()
                    continue

                try:
                    events = next(events_iterator)
                except StopIteration:
                    break

                now = time.monotonic()
                if now < next_display_time:
                    continue

                BaseFrameGenerationAlgorithm.generate_frame(events, self.frame)
                self.frame_ready.emit(bgr_frame_to_qimage(self.frame))
                next_display_time = now + self.display_period_s
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()

    @Slot(bool)
    def set_paused(self, paused):
        self.paused = paused

    @Slot()
    def stop(self):
        self.running = False
        reader = getattr(self.iterator, "reader", None)
        stream = getattr(reader, "i_events_stream", None)
        if stream is not None:
            stream.stop()


class EventViewer(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("OpenEB PySide6 Event Viewer")

        input_path = args.input_event_file or ""
        self.iterator = EventsIterator(
            input_path=input_path,
            mode="delta_t",
            delta_t=args.delta_t,
            max_duration=args.max_duration,
        )
        height, width = self.iterator.get_size()
        if width is None or height is None:
            raise RuntimeError("Could not determine sensor geometry from the event stream")

        self.paused = False
        self.last_image = None

        self.image_label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(width, height)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCentralWidget(self.image_label)

        self.shortcuts = [
            QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self.toggle_pause),
            QShortcut(QKeySequence(Qt.Key.Key_P), self, activated=self.toggle_pause),
            QShortcut(QKeySequence(Qt.Key.Key_Q), self, activated=self.close),
            QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=self.close),
        ]

        self.worker_thread = QThread(self)
        self.worker = EventReaderWorker(self.iterator, height, width, args.display_fps)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.error.connect(self.report_worker_error)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.on_worker_thread_finished)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def toggle_pause(self):
        self.paused = not self.paused
        if self.worker is not None:
            self.worker.set_paused(self.paused)

    @Slot(QImage)
    def update_frame(self, image):
        self.last_image = image
        pixmap = QPixmap.fromImage(image)
        if self.image_label.size() != pixmap.size():
            pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        self.image_label.setPixmap(pixmap)

    @Slot(str)
    def report_worker_error(self, message):
        print(f"Event reader stopped: {message}", file=sys.stderr)

    @Slot()
    def on_worker_finished(self):
        self.worker = None

    @Slot()
    def on_worker_thread_finished(self):
        self.worker_thread = None

    def resizeEvent(self, event):
        if self.last_image is not None:
            pixmap = QPixmap.fromImage(self.last_image).scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
            self.image_label.setPixmap(pixmap)
        super().resizeEvent(event)

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.stop()
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait(1000)
        super().closeEvent(event)


def main():
    args = parse_args()
    app = QApplication(sys.argv)
    try:
        viewer = EventViewer(args)
    except OSError as exc:
        if args.input_event_file:
            raise
        print(f"Failed to open a live camera: {exc}", file=sys.stderr)
        print("Run `devbox run camera-list` to check OpenEB HAL discovery.", file=sys.stderr)
        print("Run `devbox run camera-trace` for verbose HAL discovery logs.", file=sys.stderr)
        print("Run `devbox run usb-list` to check whether macOS exposes the USB device.", file=sys.stderr)
        return 1
    viewer.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
