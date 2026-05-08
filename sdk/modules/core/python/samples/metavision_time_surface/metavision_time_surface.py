# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Example of using Metavision SDK Core Python API for visualizing Time Surface of events.

This macOS-friendly version uses PySide6 for display and avoids the OpenGL-backed
metavision_sdk_ui window.
"""

import argparse
import sys

import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QSizePolicy

from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import EventPreprocessor, MostRecentTimestampBuffer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Example of using Metavision SDK Core Python API for visualizing Time Surface of events.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-i", "--input-event-file", dest="event_file_path", default="",
        help="Path to input event file (RAW or HDF5). If not specified, the camera live stream is used. "
        "If it's a camera serial number, it will try to open that camera instead.")
    parser.add_argument(
        "-j", "--input-camera-config", dest="cam_config_path", default="",
        help="Path to a JSON file containing camera config settings to restore a camera state. "
        "Only works for live cameras.")
    parser.add_argument(
        "-a", "--accumulation-time", dest="accumulation_time", type=int, default=10000,
        help="Accumulation time for which to display the Time Surface (in us).")
    args = parser.parse_args()
    if args.accumulation_time <= 0:
        parser.error("argument -a/--accumulation-time must be strictly positive")
    if args.cam_config_path and not is_live_camera(args.event_file_path):
        parser.error("argument -j/--input-camera-config only works for live cameras")
    return args


def make_events_iterator(args):
    """Create an events iterator from the input arguments."""
    if args.cam_config_path:
        from metavision_sdk_stream import Camera

        if args.event_file_path:
            camera = Camera.from_serial(args.event_file_path)
        else:
            camera = Camera.from_first_available()
        camera.load(args.cam_config_path)
        return EventsIterator.from_device(camera.get_device(), delta_t=args.accumulation_time), camera

    return EventsIterator(input_path=args.event_file_path, delta_t=args.accumulation_time), None


def rgb_frame_to_qimage(frame):
    """Return a detached QImage for a contiguous HxWx3 uint8 RGB frame."""
    contiguous = np.ascontiguousarray(frame)
    height, width, _ = contiguous.shape
    return QImage(contiguous.data, width, height, contiguous.strides[0], QImage.Format.Format_RGB888).copy()


def gray_to_jet_rgb(gray):
    """Apply a small JET-like colormap without importing OpenCV."""
    value = gray.astype(np.float32) / 255.0
    red = np.clip(1.5 - np.abs(4.0 * value - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * value - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * value - 1.0), 0.0, 1.0)
    return np.dstack((red, green, blue)).astype(np.float32).__mul__(255).astype(np.uint8)


class TimeSurfaceWorker(QObject):
    frame_ready = Signal(QImage)
    error = Signal(str)
    finished = Signal()

    def __init__(self, iterator, width, height, accumulation_time):
        super().__init__()
        self.iterator = iterator
        self.width = width
        self.height = height
        self.accumulation_time = accumulation_time
        self.running = True
        self.paused = False

    @Slot()
    def run(self):
        time_surface = MostRecentTimestampBuffer(rows=self.height, cols=self.width, channels=1)
        ts_prod = EventPreprocessor.create_TimeSurfaceProcessor(input_event_width=self.width,
                                                                input_event_height=self.height,
                                                                split_polarity=False)
        img = np.empty((self.height, self.width), dtype=np.uint8)
        try:
            for evs in self.iterator:
                if not self.running:
                    break
                while self.paused and self.running:
                    QThread.msleep(5)
                if not self.running or len(evs) == 0:
                    continue
                ts_prod.process_events(cur_frame_start_ts=evs[0][3], events_np=evs,
                                       frame_tensor_np=time_surface.numpy())
                time_surface.generate_img_time_surface(evs[-1][3], self.accumulation_time, img)
                self.frame_ready.emit(rgb_frame_to_qimage(gray_to_jet_rgb(img)))
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


class TimeSurfaceViewer(QMainWindow):
    def __init__(self, args, iterator, camera, width, height):
        super().__init__()
        self.setWindowTitle("Metavision Time Surface")
        self.camera = camera

        self.worker = None
        self.worker_thread = None
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
        self.worker = TimeSurfaceWorker(iterator, width, height, args.accumulation_time)
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
        print("Time surface stopped: {}".format(message), file=sys.stderr)

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
    try:
        iterator, camera = make_events_iterator(args)
        height, width = iterator.get_size()
        if not is_live_camera(args.event_file_path):
            iterator = LiveReplayEventsIterator(iterator)
    except OSError as exc:
        if args.event_file_path:
            raise
        print("Failed to open a live camera: {}".format(exc), file=sys.stderr)
        print("Pass an event file with `-i /path/to/recording.raw`, or check camera discovery:", file=sys.stderr)
        print("  devbox run camera-list", file=sys.stderr)
        print("  devbox run camera-trace", file=sys.stderr)
        print("  devbox run usb-list", file=sys.stderr)
        return 1
    app = QApplication(sys.argv)
    viewer = TimeSurfaceViewer(args, iterator, camera, width, height)
    viewer.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
