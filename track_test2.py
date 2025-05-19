import cv2
from mmtrack.apis import init_model        # MMTracking API :contentReference[oaicite:4]{index=4}
from mmdet.apis import inference_detector   # MMDetection API
from deep_sort_realtime.deepsort_tracker import DeepSort  # DeepSORT wrapper :contentReference[oaicite:5]{index=5}

# 2.1 Detector: load an MOT17 Faster R-CNN model with DeepSORT config
config_file = '/Users/thomaswu/Documents/GitHub/vpserver/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py'
device = 'cuda:0'

det_model = init_model(config_file, device=device)
# MMTracking unifies detection + tracking configs; this one already integrates SORT+appearance modules :contentReference[oaicite:6]{index=6}

# 2.2 Tracker: initialize DeepSORT
tracker = DeepSort(
    max_age=30,              # keep “lost” tracks for 30 frames :contentReference[oaicite:7]{index=7}
    n_init=3,                # require 3 frames to confirm a new track
    max_cosine_distance=0.2, # re-ID appearance threshold :contentReference[oaicite:8]{index=8}
)
def track_video(video_source=0):
    cap = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3.1 Person detection
        # inference_detector returns (bboxes, labels) tuples per image :contentReference[oaicite:9]{index=9}
        det_results = inference_detector(det_model, frame)
        bboxes, labels = det_results

        # 3.2 Filter for persons (COCO class 0) and format
        detections = []
        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2, score = bbox.tolist()
            if int(label) == 0 and score > 0.3:  # only person class :contentReference[oaicite:10]{index=10}
                detections.append(([x1, y1, x2, y2], score, 'person'))

        # 3.3 Update DeepSORT tracker
        tracks = tracker.update_tracks(detections, frame=frame)

        # 3.4 Visualization
        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = track.to_tlbr()  # get bbox
            track_id = track.track_id
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, f'ID {track_id}', (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow('MMTracking + DeepSORT', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    track_video('/Users/thomaswu/Downloads/5月4日16：06 闲人/DJI_20250504160604_0001_V_航点1.MP4')  # or 0 for webcam
