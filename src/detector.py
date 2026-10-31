"""detector.py - wraps the YOLO model so the rest of the project never touches Ultralytics directly."""

from ultralytics import YOLO

# In the COCO dataset (which the pretrained model was trained on), class 0 is "person".
PERSON_CLASS_ID = 0


def load_model(model_name="yolov8n.pt"):
    """Load a pretrained YOLO model.

    Input:  model_name - name of a pretrained weights file. If it is not on disk,
            Ultralytics downloads it automatically (needs internet the first time).
    Output: a YOLO model object. Load it ONCE, before the video loop, because loading is slow.
    """
    return YOLO(model_name)


def detect_people(model, frame, conf_threshold=0.4):
    """Find people in one video frame.

    Inputs:
        model          - the object returned by load_model()
        frame          - one BGR image from OpenCV (a NumPy array of shape height x width x 3)
        conf_threshold - detections with confidence below this are thrown away

    Output: a list of detections. Each detection is a tuple (x1, y1, x2, y2, confidence)
            where (x1, y1) is the top-left corner and (x2, y2) the bottom-right corner,
            in pixels. The list is empty if nobody is found.
    """
    # classes=[0] asks YOLO to report only "person". verbose=False silences per-frame console output.
    results = model(frame, classes=[PERSON_CLASS_ID], conf=conf_threshold, verbose=False)

    # We passed one image, so results has exactly one element.
    result = results[0]

    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # corner coordinates as floats
        confidence = float(box.conf[0])
        detections.append((int(x1), int(y1), int(x2), int(y2), confidence))
    return detections
