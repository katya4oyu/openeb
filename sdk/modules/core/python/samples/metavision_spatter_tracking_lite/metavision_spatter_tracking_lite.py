# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Lightweight spatter tracking sample using only open Metavision Core Python APIs.

The sample groups events in a coarse grid, extracts connected active cells as
small moving objects, and assigns stable IDs by nearest-neighbor matching.
"""

import argparse
import colorsys
import math
import sys
import time

import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QSizePolicy

from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Metavision lightweight spatter tracking sample.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-i", "--input-event-file", dest="event_file_path", default="",
        help="Path to input event file (RAW, DAT or HDF5). If not specified, the camera live stream is used. "
        "If it's a camera serial number, it will try to open that camera instead.")
    parser.add_argument("--process-from", dest="process_from", type=int, default=0,
                        help="Start timestamp, in us.")
    parser.add_argument("--process-to", dest="process_to", type=int, default=None,
                        help="End timestamp, in us. Omit to process until the end.")
    parser.add_argument(
        "-r", "--replay-factor", dest="replay_factor", type=float, default=1.0,
        help="Replay factor for event files. Values greater than 1.0 replay slower than real time.")
    parser.add_argument("--accumulation-time-us", dest="accumulation_time_us", type=int, default=50,
                        help="Time slice duration used for detection and display.")
    parser.add_argument("--cell-width", dest="cell_width", type=int, default=4,
                        help="Grid cell width, in pixels.")
    parser.add_argument("--cell-height", dest="cell_height", type=int, default=4,
                        help="Grid cell height, in pixels.")
    parser.add_argument("--activation-ths", dest="activation_ths", type=int, default=2,
                        help="Minimum number of events in a cell to activate it.")
    parser.add_argument("--min-size", dest="min_size", type=int, default=2,
                        help="Minimum cluster width or height, in pixels.")
    parser.add_argument("--max-size", dest="max_size", type=int, default=80,
                        help="Maximum cluster width and height, in pixels.")
    parser.add_argument("--bbox-padding", dest="bbox_padding", type=int, default=2,
                        help="Padding added around event-derived boxes, in pixels.")
    parser.add_argument("--bbox-quantile", dest="bbox_quantile", type=float, default=0.05,
                        help="Lower/upper event coordinate quantile trimmed when estimating box size.")
    parser.add_argument("--min-events-per-object", dest="min_events_per_object", type=int, default=2,
                        help="Minimum number of events required to keep a detected object.")
    parser.add_argument("--max-distance", dest="max_distance", type=float, default=24.0,
                        help="Maximum center-to-center distance used to associate clusters across slices.")
    parser.add_argument("--max-missed", dest="max_missed", type=int, default=3,
                        help="Number of consecutive unmatched slices before a track is removed.")
    parser.add_argument("--display-fps", dest="display_fps", type=float, default=30.0,
                        help="Maximum Qt display refresh rate.")
    parser.add_argument("--output-video", dest="output_video", default="",
                        help="Optional output video path.")
    return parser.parse_args()


def validate_args(args):
    """Validate user parameters before opening the input stream."""
    if args.process_to is not None and args.process_to <= args.process_from:
        raise ValueError("--process-to must be greater than --process-from")
    if args.accumulation_time_us <= 0:
        raise ValueError("--accumulation-time-us must be strictly positive")
    if args.cell_width <= 0 or args.cell_height <= 0:
        raise ValueError("--cell-width and --cell-height must be strictly positive")
    if args.activation_ths <= 0:
        raise ValueError("--activation-ths must be strictly positive")
    if args.min_size <= 0 or args.max_size < args.min_size:
        raise ValueError("--max-size must be greater than or equal to --min-size")
    if args.bbox_padding < 0:
        raise ValueError("--bbox-padding must be positive or zero")
    if not 0.0 <= args.bbox_quantile < 0.5:
        raise ValueError("--bbox-quantile must be in the [0, 0.5) range")
    if args.min_events_per_object <= 0:
        raise ValueError("--min-events-per-object must be strictly positive")
    if args.max_distance <= 0:
        raise ValueError("--max-distance must be strictly positive")
    if args.max_missed < 0:
        raise ValueError("--max-missed must be positive or zero")
    if args.display_fps <= 0:
        raise ValueError("--display-fps must be strictly positive")


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


def id_color(track_id):
    """Return a stable RGB color for a track ID."""
    hue = ((track_id * 37) % 360) / 360.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(255 * red), int(255 * green), int(255 * blue)


class LightweightSpatterTracker:
    """Tracks small non-colliding moving event clusters."""

    def __init__(self, width, height, cell_width, cell_height, activation_ths, min_size, max_size, max_distance,
                 max_missed, bbox_padding=2, bbox_quantile=0.05, min_events_per_object=2):
        self.width = width
        self.height = height
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.activation_ths = activation_ths
        self.min_size = min_size
        self.max_size = max_size
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.bbox_padding = bbox_padding
        self.bbox_quantile = bbox_quantile
        self.min_events_per_object = min_events_per_object
        self.grid_width = int(math.ceil(width / float(cell_width)))
        self.grid_height = int(math.ceil(height / float(cell_height)))
        self.tracks = {}
        self.next_track_id = 1

    def process_events(self, evs, ts):
        """Detect clusters in a slice and update tracks."""
        detections = self._detect(evs)
        return self._update_tracks(detections, ts)

    def _detect(self, evs):
        if len(evs) == 0:
            return []

        grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint16)
        xs = np.minimum(evs["x"].astype(np.int32) // self.cell_width, self.grid_width - 1)
        ys = np.minimum(evs["y"].astype(np.int32) // self.cell_height, self.grid_height - 1)
        np.add.at(grid, (ys, xs), 1)

        active = (grid >= self.activation_ths).astype(np.uint8)
        detections = []
        for component in self._connected_components(active):
            x_cell, y_cell, w_cell, h_cell, active_cells = component
            mask = self._events_in_component(evs, component)
            event_count = int(mask.sum())
            if event_count < self.min_events_per_object:
                continue

            x, y, w, h = self._event_bbox(evs["x"][mask], evs["y"][mask])
            if max(w, h) < self.min_size or w > self.max_size or h > self.max_size:
                continue

            detections.append({
                "bbox": (x, y, w, h),
                "center": (x + 0.5 * w, y + 0.5 * h),
                "active_cells": int(active_cells),
                "event_count": event_count,
            })
        return detections

    def _events_in_component(self, evs, component):
        x_cell, y_cell, w_cell, h_cell, _ = component
        x0 = x_cell * self.cell_width
        y0 = y_cell * self.cell_height
        x1 = min(self.width, (x_cell + w_cell) * self.cell_width)
        y1 = min(self.height, (y_cell + h_cell) * self.cell_height)
        return (evs["x"] >= x0) & (evs["x"] < x1) & (evs["y"] >= y0) & (evs["y"] < y1)

    def _event_bbox(self, xs, ys):
        if self.bbox_quantile > 0 and len(xs) >= 4:
            low = self.bbox_quantile
            high = 1.0 - self.bbox_quantile
            x0 = int(math.floor(np.quantile(xs, low)))
            y0 = int(math.floor(np.quantile(ys, low)))
            x1 = int(math.ceil(np.quantile(xs, high)))
            y1 = int(math.ceil(np.quantile(ys, high)))
        else:
            x0 = int(xs.min())
            y0 = int(ys.min())
            x1 = int(xs.max())
            y1 = int(ys.max())

        x0 = max(0, x0 - self.bbox_padding)
        y0 = max(0, y0 - self.bbox_padding)
        x1 = min(self.width - 1, x1 + self.bbox_padding)
        y1 = min(self.height - 1, y1 + self.bbox_padding)
        return x0, y0, x1 - x0 + 1, y1 - y0 + 1

    @staticmethod
    def _connected_components(active):
        visited = np.zeros(active.shape, dtype=bool)
        height, width = active.shape
        components = []

        for y_start, x_start in zip(*np.nonzero(active)):
            if visited[y_start, x_start]:
                continue
            stack = [(int(x_start), int(y_start))]
            visited[y_start, x_start] = True
            min_x = max_x = int(x_start)
            min_y = max_y = int(y_start)
            count = 0

            while stack:
                x, y = stack.pop()
                count += 1
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                for ny in range(max(0, y - 1), min(height, y + 2)):
                    for nx in range(max(0, x - 1), min(width, x + 2)):
                        if visited[ny, nx] or not active[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((nx, ny))

            components.append((min_x, min_y, max_x - min_x + 1, max_y - min_y + 1, count))
        return components

    def _update_tracks(self, detections, ts):
        track_ids = list(self.tracks.keys())
        pairs = []
        for det_idx, det in enumerate(detections):
            dx, dy = det["center"]
            for track_id in track_ids:
                tx, ty = self.tracks[track_id]["center"]
                distance = math.hypot(dx - tx, dy - ty)
                if distance <= self.max_distance:
                    pairs.append((distance, det_idx, track_id))

        matched_detections = set()
        matched_tracks = set()
        for _, det_idx, track_id in sorted(pairs):
            if det_idx in matched_detections or track_id in matched_tracks:
                continue
            self._assign_detection(track_id, detections[det_idx], ts)
            matched_detections.add(det_idx)
            matched_tracks.add(track_id)

        for det_idx, det in enumerate(detections):
            if det_idx in matched_detections:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id] = self._make_track(track_id, det, ts)
            matched_tracks.add(track_id)

        for track_id in list(self.tracks.keys()):
            if track_id in matched_tracks:
                continue
            self.tracks[track_id]["missed"] += 1
            if self.tracks[track_id]["missed"] > self.max_missed:
                del self.tracks[track_id]

        return [track for track in self.tracks.values() if track["missed"] == 0]

    def _assign_detection(self, track_id, detection, ts):
        track = self.tracks[track_id]
        track["bbox"] = detection["bbox"]
        track["center"] = detection["center"]
        track["active_cells"] = detection["active_cells"]
        track["event_count"] = detection["event_count"]
        track["last_ts"] = ts
        track["missed"] = 0

    def _make_track(self, track_id, detection, ts):
        return {
            "id": track_id,
            "bbox": detection["bbox"],
            "center": detection["center"],
            "active_cells": detection["active_cells"],
            "event_count": detection["event_count"],
            "first_ts": ts,
            "last_ts": ts,
            "missed": 0,
            "color": id_color(track_id),
        }


def draw_tracking_results(image, tracks):
    """Draw bounding boxes and track IDs over a QImage."""
    painter = QPainter(image)
    painter.setFont(QFont("Helvetica", 10))
    for track in tracks:
        x, y, w, h = track["bbox"]
        color = QColor(*track["color"])
        painter.setPen(QPen(color, 1))
        painter.drawRect(x, y, max(1, w), max(1, h))
        painter.drawEllipse(int(track["center"][0]) - 2, int(track["center"][1]) - 2, 4, 4)
        painter.drawText(x, max(10, y - 3), "{} {}x{} e{}".format(track["id"], w, h, track["event_count"]))
    painter.end()


def make_video_writer(path, width, height):
    """Create a best-effort OpenCV video writer."""
    if not path:
        return None
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, 25.0, (width, height))


class SpatterTrackingWorker(QObject):
    """Runs event reading and tracking off the Qt UI thread."""

    frame_ready = Signal(QImage)
    error = Signal(str)
    finished = Signal()

    def __init__(self, iterator, width, height, args):
        super().__init__()
        self.iterator = iterator
        self.width = width
        self.height = height
        self.args = args
        self.display_period_s = 1.0 / args.display_fps
        self.paused = False
        self.running = True

    @Slot()
    def run(self):
        frame_gen = OnDemandFrameGenerationAlgorithm(self.width, self.height,
                                                     accumulation_time_us=self.args.accumulation_time_us)
        tracker = LightweightSpatterTracker(self.width, self.height, self.args.cell_width, self.args.cell_height,
                                            self.args.activation_ths, self.args.min_size, self.args.max_size,
                                            self.args.max_distance, self.args.max_missed, self.args.bbox_padding,
                                            self.args.bbox_quantile, self.args.min_events_per_object)
        output_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        video_writer = make_video_writer(self.args.output_video, self.width, self.height)
        next_display_time = 0.0

        try:
            for evs in self.iterator:
                if not self.running:
                    break
                while self.paused and self.running:
                    QThread.msleep(5)
                    next_display_time = time.monotonic()
                if not self.running:
                    break

                ts = self.iterator.get_current_time()
                frame_gen.process_events(evs)
                frame_gen.generate(ts, output_img)
                tracks = tracker.process_events(evs, ts)
                image = bgr_frame_to_qimage(output_img)
                draw_tracking_results(image, tracks)

                if video_writer is not None:
                    video_writer.write(output_img)

                now = time.monotonic()
                if now >= next_display_time:
                    self.frame_ready.emit(image)
                    next_display_time = now + self.display_period_s
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if video_writer is not None:
                video_writer.release()
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


class SpatterTrackingViewer(QMainWindow):
    """PySide6 viewer for lightweight spatter tracking."""

    def __init__(self, args, iterator, width, height):
        super().__init__()
        self.setWindowTitle("Metavision Spatter Tracking Lite")
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
        self.worker = SpatterTrackingWorker(iterator, width, height, args)
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
        print("Spatter tracking stopped: {}".format(message), file=sys.stderr)

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
    """Main."""
    args = parse_args()
    validate_args(args)

    try:
        max_duration = args.process_to - args.process_from if args.process_to is not None else None
        iterator = EventsIterator(input_path=args.event_file_path, start_ts=args.process_from,
                                  max_duration=max_duration, delta_t=args.accumulation_time_us)
        if args.replay_factor > 0 and not is_live_camera(args.event_file_path):
            iterator = LiveReplayEventsIterator(iterator, replay_factor=args.replay_factor)
        height, width = iterator.get_size()
    except OSError as exc:
        if args.event_file_path:
            raise
        print("Failed to open a live camera: {}".format(exc), file=sys.stderr)
        print("Run `devbox run camera-list` to check OpenEB HAL discovery.", file=sys.stderr)
        print("Run `devbox run camera-trace` for verbose HAL discovery logs.", file=sys.stderr)
        print("Run `devbox run usb-list` to check whether macOS exposes the USB device.", file=sys.stderr)
        return 1
    print("Metavision lightweight spatter tracking sample.")
    print("Press Space/P to pause, Q/Escape to quit.")
    app = QApplication(sys.argv)
    viewer = SpatterTrackingViewer(args, iterator, width, height)
    viewer.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
