# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Example of using Metavision SDK API for building a passive radar for moving objects.

This macOS-friendly Python version uses PySide6 for display.
"""

import argparse
import math
import sys

import numpy as np
from PySide6.QtCore import QObject, QPointF, QThread, Qt, Signal, Slot
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QSizePolicy

from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Metavision Dummy Radar sample.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-i", "--input-event-file", dest="event_file_path", default="",
        help="Path to input event file (RAW or HDF5). If not specified, the camera live stream is used. "
        "If it's a camera serial number, it will try to open that camera instead.")
    parser.add_argument("-a", "--accumulation-time", dest="delta_t", type=int, default=50000,
                        help="Accumulation time, in us, used to compute each radar slice.")
    parser.add_argument("--min-ev-rate", dest="min_ev_rate", type=float, default=1e5,
                        help="Minimum event rate per bin displayed in the radar view.")
    parser.add_argument("--max-ev-rate", dest="max_ev_rate", type=float, default=3e6,
                        help="Maximum event rate per bin used to scale the radar view.")
    parser.add_argument("--cam-fov-deg", dest="camera_fov", type=float, default=90.0,
                        help="Camera lateral field of view, in degrees.")
    parser.add_argument("--nbins", dest="nbins", type=int, default=8,
                        help="Number of bins describing the sensor width.")
    parser.add_argument("-r", "--replay-factor", dest="replay_factor", type=float, default=1.0,
                        help="Replay factor for event files. Values greater than 1.0 replay slower than real time.")
    return parser.parse_args()


def validate_args(args):
    if args.delta_t <= 0:
        raise ValueError("The accumulation time must be strictly positive.")
    if args.nbins <= 0:
        raise ValueError("The number of bins must be strictly positive.")
    if args.min_ev_rate < 0:
        raise ValueError("The minimum event rate must be positive or zero.")
    if args.max_ev_rate <= args.min_ev_rate:
        raise ValueError("The maximum event rate must be greater than the minimum event rate.")
    if args.camera_fov <= 0:
        raise ValueError("The camera lateral field of view must be strictly positive.")


def compute_event_rates(evs, width, nbins, delta_t):
    if evs.size == 0:
        return np.zeros(nbins, dtype=np.float32)

    bin_indices = (evs["x"].astype(np.uint32) * nbins // width).astype(np.int32)
    np.minimum(bin_indices, nbins - 1, out=bin_indices)
    counts = np.bincount(bin_indices, minlength=nbins)
    return counts.astype(np.float32) / (delta_t * 1e-6)


def bgr_frame_to_qimage(frame):
    """Return a detached QImage for a contiguous HxWx3 uint8 BGR/RGB frame."""
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


def draw_radar(painter, x_offset, width, height, rates, args):
    painter.fillRect(x_offset, 0, width, height, QColor(0, 0, 0))
    center = QPointF(x_offset + width / 2.0, float(height))
    fov = math.radians(args.camera_fov)
    radius_max = float(height)
    grid_pen = QPen(QColor(0, 180, 0), 1)
    painter.setPen(grid_pen)

    for ring in range(1, 6):
        radius = radius_max * ring / 5.0
        painter.drawArc(int(center.x() - radius), int(center.y() - radius), int(2 * radius), int(2 * radius),
                        int((90 - args.camera_fov / 2.0) * 16), int(args.camera_fov * 16))

    for bin_idx in range(args.nbins + 1):
        theta = -fov / 2.0 + fov * bin_idx / args.nbins
        end = QPointF(center.x() + radius_max * math.sin(theta), center.y() - radius_max * math.cos(theta))
        painter.drawLine(center, end)

    if rates.size == 0:
        return

    max_bin = int(np.argmax(rates))
    denom = args.max_ev_rate - args.min_ev_rate
    for bin_idx, rate in enumerate(rates):
        if rate < args.min_ev_rate:
            continue
        ratio = float(np.clip((rate - args.min_ev_rate) / denom, 0.0, 1.0))
        radius = radius_max * ratio
        theta0 = -fov / 2.0 + fov * bin_idx / args.nbins
        theta1 = -fov / 2.0 + fov * (bin_idx + 1) / args.nbins
        theta = 0.5 * (theta0 + theta1)
        point = QPointF(center.x() + radius * math.sin(theta), center.y() - radius * math.cos(theta))
        color = QColor(255, 255, 255) if bin_idx == max_bin else QColor(255, 180, 0)
        painter.setPen(QPen(color, 2))
        painter.drawLine(center, point)
        painter.drawEllipse(point, 4, 4)


def compose_frame(event_frame, rates, args):
    height, width = event_frame.shape[:2]
    event_image = bgr_frame_to_qimage(event_frame)
    image = QImage(width * 2, height, QImage.Format.Format_RGB888)
    image.fill(QColor(0, 0, 0))
    painter = QPainter(image)
    painter.drawImage(0, 0, event_image)
    draw_radar(painter, width, width, height, rates, args)
    painter.end()
    return image


class DummyRadarWorker(QObject):
    frame_ready = Signal(QImage)
    error = Signal(str)
    finished = Signal()

    def __init__(self, iterator, width, height, args):
        super().__init__()
        self.iterator = iterator
        self.width = width
        self.height = height
        self.args = args
        self.running = True
        self.paused = False

    @Slot()
    def run(self):
        frame_gen = OnDemandFrameGenerationAlgorithm(self.width, self.height, accumulation_time_us=self.args.delta_t)
        event_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        try:
            for evs in self.iterator:
                if not self.running:
                    break
                while self.paused and self.running:
                    QThread.msleep(5)
                if not self.running:
                    break

                frame_gen.process_events(evs)
                ts = self.iterator.get_current_time()
                frame_gen.generate(ts, event_frame)
                rates = compute_event_rates(evs, self.width, self.args.nbins, self.args.delta_t)
                self.frame_ready.emit(compose_frame(event_frame, rates, self.args))
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
        iterator = getattr(self.iterator, "iterator", self.iterator)
        reader = getattr(iterator, "reader", None)
        stream = getattr(reader, "i_events_stream", None)
        if stream is not None:
            stream.stop()


class DummyRadarViewer(QMainWindow):
    def __init__(self, args, iterator, width, height):
        super().__init__()
        self.setWindowTitle("Metavision Dummy Radar")

        self.worker = None
        self.worker_thread = None
        self.paused = False
        self.last_image = None

        self.image_label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(2 * width, height)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCentralWidget(self.image_label)

        self.shortcuts = [
            QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self.toggle_pause),
            QShortcut(QKeySequence(Qt.Key.Key_P), self, activated=self.toggle_pause),
            QShortcut(QKeySequence(Qt.Key.Key_Q), self, activated=self.close),
            QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=self.close),
        ]

        self.worker_thread = QThread(self)
        self.worker = DummyRadarWorker(iterator, width, height, args)
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
            pixmap = pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.FastTransformation)
        self.image_label.setPixmap(pixmap)

    @Slot(str)
    def report_worker_error(self, message):
        print("Dummy radar stopped: {}".format(message), file=sys.stderr)

    @Slot()
    def on_worker_finished(self):
        self.worker = None

    @Slot()
    def on_worker_thread_finished(self):
        self.worker_thread = None

    def resizeEvent(self, event):
        if self.last_image is not None:
            pixmap = QPixmap.fromImage(self.last_image).scaled(self.image_label.size(),
                                                               Qt.AspectRatioMode.KeepAspectRatio,
                                                               Qt.TransformationMode.FastTransformation)
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
    validate_args(args)
    try:
        iterator = EventsIterator(input_path=args.event_file_path, delta_t=args.delta_t)
        if args.replay_factor > 0 and not is_live_camera(args.event_file_path):
            iterator = LiveReplayEventsIterator(iterator, replay_factor=args.replay_factor)
        height, width = iterator.get_size()
    except OSError as exc:
        if args.event_file_path:
            raise
        print("Failed to open a live camera: {}".format(exc), file=sys.stderr)
        print("Pass an event file with `-i /path/to/recording.raw`, or check camera discovery:", file=sys.stderr)
        print("  devbox run camera-list", file=sys.stderr)
        print("  devbox run camera-trace", file=sys.stderr)
        print("  devbox run usb-list", file=sys.stderr)
        return 1
    print("Metavision Dummy Radar sample.")
    print("Press Space/P to pause, Q/Escape to quit.")
    app = QApplication(sys.argv)
    viewer = DummyRadarViewer(args, iterator, width, height)
    viewer.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
