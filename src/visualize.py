"""visualize.py - small helpers that draw on a frame. They only draw; they do no detection or tracking."""

import cv2

GREEN = (0, 255, 0)  # OpenCV colours are (Blue, Green, Red)


def draw_detections(frame, detections):
    """Draw a rectangle and a confidence label for each detection.

    Inputs:  frame - BGR image (modified in place)
             detections - list of (x1, y1, x2, y2, confidence)
    Output:  None (the frame itself is changed)
    """
    for x1, y1, x2, y2, confidence in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
        label = f"person {confidence:.2f}"
        # Put the text just above the box, but keep it on screen if the box touches the top edge.
        text_y = max(y1 - 8, 15)
        cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)


def draw_fps(frame, fps):
    """Draw the FPS number in the top-left corner.

    Inputs:  frame - BGR image (modified in place); fps - a number
    Output:  None
    """
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
