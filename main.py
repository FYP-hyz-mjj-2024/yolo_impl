import cv2
from ultralytics import YOLO
import torch

model = YOLO("yolov8s.pt")


capture = cv2.VideoCapture("surfers.mp4")   # Video capture obj

# Plays video frame by frame
while True:
    # Read frame
    is_frame, this_frame = capture.read()
    if not is_frame:
        break

    # Detect objects at this frame
    if torch.backends.mps.is_available():
        results = model(this_frame, device="mps")
    else:
        results = model(this_frame)
    print(results)


    # Show frames, wait for interrupt
    cv2.imshow("Video", this_frame)
    if cv2.waitKey(1) == 27:
        break

capture.release()
cv2.destroyAllWindows()