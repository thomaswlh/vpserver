# detect_and_track.py

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def initialize_detector(model_name="./weights/yolo11n.pt"):
    """
    Load the YOLOv5 model. 
    The 'yolov5s.pt' is a small, fast model; you can swap for 'yolov5m.pt' or custom weights.
    """
    model = YOLO(model_name)  # Ultralytics API :contentReference[oaicite:5]{index=5}
    return model

def initialize_tracker(max_age=30, n_init=3, max_cosine_distance=0.3):
    """
    Initialize the DeepSORT tracker with motion & appearance settings.
    - max_age: how many frames to keep "lost" tracks
    - n_init: detections before confirming a new track
    - max_cosine_distance: appearance threshold for re-ID
    """
    return DeepSort(max_age=max_age,
                    n_init=n_init,
                    max_cosine_distance=max_cosine_distance)

def detect_people(model, frame):
    """
    Run YOLOv5 on the frame, filter detections for 'person' class (class 0 in COCO).
    Returns a list of [x1,y1,x2,y2,confidence] for each person.
    """
    results = model(frame,conf=0.7)[0]  # single-frame inference :contentReference[oaicite:6]{index=6}
    persons = []
    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = r
        if int(cls) == 0 and conf > 0.3:  # class 0 = person, confidence threshold
            persons.append(([x1, y1, x2, y2], conf, 'person'))
    return persons

def run_video(source=0):
    """
    Capture from webcam (source=0) or video file.  
    Displays real-time tracking with bounding boxes and IDs.
    """
    cap = cv2.VideoCapture(source)
    detector = initialize_detector()
    tracker = initialize_tracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect people
        detections = detect_people(detector, frame)

        # Update tracker: DeepSort expects detections as (bbox, confidence, class)
        tracks = tracker.update_tracks(detections, frame=frame)

        # Draw tracks
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = track.to_tlbr()  # top-left, bottom-right
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, f'ID {track_id}', (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow("YOLOv5 + DeepSORT Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_video("/Users/thomaswu/Downloads/5月4日16：06 闲人/DJI_20250504160604_0001_V_航点1.MP4")  # change to "video.mp4" to process a file
