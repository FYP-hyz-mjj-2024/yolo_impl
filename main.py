import cv2
from ultralytics import YOLO
import torch

model = YOLO("yolov8s.pt")

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH,600)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,500)

# Lazy model func. 
# Only need to determine mps availability once
model_ = lambda frame: model(frame, device="mps") if torch.backends.mps.is_available else model(frame)

# Plays video frame by frame
while True:
    # Read frame
    is_frame, this_frame = capture.read()
    if not is_frame:
        break

    # Detect objects at this frame
    results = model_(this_frame)
    
    for result in results:
        bboxes = result.boxes.xyxy.cpu().numpy().astype("int")  # Coordinates of boxes
        classes = result.boxes.cls.cpu().numpy().astype("int")  # Classes (index) of boxes
        confidences = result.boxes.conf.cpu().numpy()   # Confidences of boxes

        # Draw box on detected objects
        for i in range(len(bboxes)):
            # Draw box
            (x1, y1, x2, y2) = bboxes[i]
            cv2.rectangle(this_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label & Possibility
            label = f"{model.names[classes[i]]} {confidences[i]:.2f}" #{result.boxes.conf[cls_idx]:.2f}
            cv2.putText(this_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, color=(0, 255, 0), thickness=2)


    # Show frames, wait for interrupt
    cv2.imshow("Video", this_frame)
    if cv2.waitKey(1) == 27:
        break

capture.release()
cv2.destroyAllWindows()