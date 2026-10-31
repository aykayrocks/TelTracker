r"""main.py - entry point. Reads a video, detects people in each frame, draws results, shows a window.

Run from the project root:
    python src\main.py --source videos\sample.mp4
Press 'q' in the video window to quit.
"""

import argparse
import sys
import time

import cv2

from detector import load_model, detect_people
from visualize import draw_detections, draw_fps


def parse_args():
    parser = argparse.ArgumentParser(description="Detect people in a video with YOLO.")
    parser.add_argument("--source", required=True, help="path to a video file")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO weights file (default: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.4, help="minimum confidence (default: 0.4)")
    return parser.parse_args()


def main():
    args = parse_args()

    model = load_model(args.model)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: could not open video '{args.source}'. Check the path.")
        sys.exit(1)

    smoothed_fps = 0.0
    previous_time = time.perf_counter()

    while True:
        ok, frame = cap.read()  # ok is False when the video ends (or a read fails)
        if not ok:
            print("End of video.")
            break

        detections = detect_people(model, frame, args.conf)
        draw_detections(frame, detections)

        # FPS = 1 / (time taken for this loop iteration). Raw values jump around,
        # so we blend 90% of the old value with 10% of the new one to keep the number readable.
        now = time.perf_counter()
        current_fps = 1.0 / (now - previous_time)
        previous_time = now
        smoothed_fps = current_fps if smoothed_fps == 0.0 else 0.9 * smoothed_fps + 0.1 * current_fps
        draw_fps(frame, smoothed_fps)

        cv2.imshow("People detection", frame)
        if cv2.waitKey(100) & 0xFF == ord("q"):  # waitKey also lets the window refresh
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
