#!/usr/bin/env python3
"""Detect faces in a webcam stream, or in a still image.

Live window: press Q or Esc to quit, or close the window.
No webcam:  python face_detect.py --image samples/nasa-astronaut-group-8.jpg
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

WINDOW_NAME = "Face Detection"
MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"

# YuNet is reliable up to a high-definition frame. Larger photos are
# scaled down for detection, and the boxes are mapped back afterwards.
PRIMARY_MAX_SIDE = 1920
# A face that fills a high-resolution portrait is larger than the sizes
# YuNet was trained on, so a failed first pass is retried smaller.
FALLBACK_MAX_SIDE = 960

SCORE_THRESHOLD = 0.8
NMS_THRESHOLD = 0.3
BOX_COLOUR = (70, 220, 90)  # BGR

# Named log levels are not exported by every wheel. 1 keeps fatal messages
# and hides the noisy "camera index out of range" error when nothing is plugged in.
OPENCV_LOG_FATAL = 1


def project_dir() -> Path:
    return Path(__file__).resolve().parent


def model_path() -> Path:
    return project_dir() / "models" / MODEL_FILENAME


def create_detector() -> cv2.FaceDetectorYN:
    path = model_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Face detection model not found:\n  {path}\n"
            "Keep the models folder next to face_detect.py."
        )
    # The input size is updated for every frame in detect_faces.
    return cv2.FaceDetectorYN.create(
        str(path),
        "",
        (320, 320),
        SCORE_THRESHOLD,
        NMS_THRESHOLD,
        5000,
    )


def _boxes_at_scale(image, detector, max_side: int) -> list[tuple[int, int, int, int]]:
    height, width = image.shape[:2]
    working = image
    long_side = max(height, width)
    if long_side > max_side:
        scale = max_side / long_side
        new_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        working = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    detect_height, detect_width = working.shape[:2]
    detector.setInputSize((detect_width, detect_height))
    _, faces = detector.detect(working)
    if faces is None:
        return []

    scale_x = width / detect_width
    scale_y = height / detect_height
    boxes = []
    for face in faces:
        x = int(round(float(face[0]) * scale_x))
        y = int(round(float(face[1]) * scale_y))
        w = int(round(float(face[2]) * scale_x))
        h = int(round(float(face[3]) * scale_y))
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        boxes.append((x, y, w, h))
    return boxes


def detect_faces(image, detector) -> list[tuple[int, int, int, int]]:
    """Return each face as (x, y, width, height) in image pixels."""
    boxes = _boxes_at_scale(image, detector, PRIMARY_MAX_SIDE)
    long_side = max(image.shape[:2])
    if not boxes and long_side > FALLBACK_MAX_SIDE:
        boxes = _boxes_at_scale(image, detector, FALLBACK_MAX_SIDE)
    return boxes


def _draw_badge(frame, text: str, origin: tuple[int, int], anchor: str = "top") -> None:
    x, y = origin
    height = frame.shape[0]
    scale = max(0.55, min(1.05, height / 700))
    thickness = 2 if scale >= 0.8 else 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad_x, pad_y = 10, 8
    box_width = text_width + pad_x * 2
    box_height = text_height + baseline + pad_y * 2
    top = y - box_height if anchor == "bottom" else y
    cv2.rectangle(
        frame,
        (x, top),
        (x + box_width, top + box_height),
        (28, 28, 28),
        thickness=-1,
    )
    cv2.putText(
        frame,
        text,
        (x + pad_x, top + pad_y + text_height),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def annotate(frame, boxes, fps: float | None = None):
    """Draw face boxes, the face count, and (in live mode) the frame rate."""
    height = frame.shape[0]
    thickness = max(2, round(height / 260))
    for x, y, w, h in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOUR, thickness)

    count = len(boxes)
    label = "1 face" if count == 1 else f"{count} faces"
    if fps is not None:
        label = f"{label}    {fps:.0f} FPS"
    _draw_badge(frame, label, (16, 16))
    if fps is not None:
        _draw_badge(frame, "Q or Esc to quit", (16, height - 16), anchor="bottom")
    return frame


def _print_no_camera(index: int) -> None:
    print(
        f"No camera was found (index {index}).\n\n"
        "Check that a webcam is plugged in, that the operating system can see it,\n"
        "and that another application is not already using it. Try another index\n"
        "with --camera 1.\n\n"
        "To try the detector without a webcam, run:\n\n"
        "    python face_detect.py --image samples/nasa-astronaut-group-8.jpg",
        file=sys.stderr,
    )


def _window_closed() -> bool:
    try:
        visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
    except cv2.error:
        return True
    return visible < 1


def run_camera(index: int) -> int:
    detector = create_detector()
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        camera.release()
        _print_no_camera(index)
        return 1

    # A device can report itself as open and still never deliver a frame.
    ok, frame = camera.read()
    if not ok or frame is None:
        camera.release()
        _print_no_camera(index)
        return 1

    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("Camera started. Press Q or Esc to quit, or close the window.")
    fps = 0.0
    previous = time.perf_counter()
    try:
        while True:
            boxes = detect_faces(frame, detector)
            shown_fps = fps if fps > 0 else None
            cv2.imshow(WINDOW_NAME, annotate(frame, boxes, fps=shown_fps))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("Quitting.")
                break
            if _window_closed():
                break

            ok, frame = camera.read()
            if not ok or frame is None:
                print("The camera stopped sending frames.", file=sys.stderr)
                return 1

            now = time.perf_counter()
            elapsed = now - previous
            previous = now
            if elapsed > 0:
                instant = 1.0 / elapsed
                fps = instant if fps == 0 else (0.9 * fps + 0.1 * instant)
    except KeyboardInterrupt:
        print("\nQuitting.")
    except cv2.error:
        print(
            "OpenCV could not open a display window. On a machine without a\n"
            "screen, run the detector on a photo instead:\n\n"
            "    python face_detect.py --image samples/nasa-astronaut-group-8.jpg",
            file=sys.stderr,
        )
        return 1
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


def run_image(path: Path, output: Path | None, show: bool) -> int:
    image = cv2.imread(str(path))
    if image is None:
        print(f"Could not read the image file: {path}", file=sys.stderr)
        return 1

    boxes = detect_faces(image, create_detector())
    annotated = annotate(image, boxes)
    destination = output or path.with_name(f"{path.stem}_detected{path.suffix}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), annotated):
        print(f"Could not save the result to: {destination}", file=sys.stderr)
        return 1

    count = len(boxes)
    noun = "face" if count == 1 else "faces"
    print(f"Detected {count} {noun}.")
    print(f"Saved the annotated image to {destination}")

    if not show:
        return 0

    try:
        cv2.imshow(WINDOW_NAME, annotated)
        print("Press any key in the image window to close it.")
        cv2.waitKey(0)
    except cv2.error:
        print(
            "Saved the image, but could not open a window on this machine.",
            file=sys.stderr,
        )
        return 1
    finally:
        cv2.destroyAllWindows()
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect faces from the webcam, or from a still image.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Detect faces in this image instead of using the webcam.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Where to save the annotated image (default: next to the input).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index to open (default: 0).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display the annotated image. Only used with --image.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.output is not None and args.image is None:
        print("--output is only used together with --image.", file=sys.stderr)
        return 1
    if args.show and args.image is None:
        print(
            "--show is only used together with --image. The webcam is always shown.",
            file=sys.stderr,
        )
        return 1

    cv2.setLogLevel(OPENCV_LOG_FATAL)
    try:
        if args.image is not None:
            if not args.image.is_file():
                print(f"Image not found: {args.image}", file=sys.stderr)
                return 1
            return run_image(args.image, args.output, args.show)
        return run_camera(args.camera)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
