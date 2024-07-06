import cv2
from ultralytics import YOLO
import torch

model = YOLO("yolov8s.pt")

# Lazy model func. 
if torch.cuda.is_available():
    model.model = model.model.half()  # Convert model to half precision for CUDA
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'


capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

# Plays video frame by frame
while True:
    # Read frame
    is_frame, this_frame = capture.read()
    if not is_frame:
        break

    # Detect objects at this frame
    results = model(this_frame, device=device)

    # Note: results only have one element here, which is the set of objects in this frame.
    # Unless the input has multiple images (i.e., this_frame is multiple images)
    result = results[0]

    # Note: results only have one element here, which is the set of objects in this frame.
    # for result in results:    No need to use a for loop
    bboxes = result.boxes.xyxy.cpu().numpy().astype("int")  # Coordinates of boxes
    classes = result.boxes.cls.cpu().numpy().astype("int")  # Classes (index) of boxes
    confidences = result.boxes.conf.cpu().numpy()           # Confidences of boxes

    # Draw box on detected objects
    for box, class_id, confidence in zip(bboxes, classes, confidences):
        # Draw box
        x1, y1, x2, y2 = box
        cv2.rectangle(this_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw text
        label = f"{model.names[class_id]} {confidence:.2f}"
        cv2.putText(this_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show frames, wait for interrupt
    cv2.imshow("Video", this_frame)
    if cv2.waitKey(1) == 27:
        break

capture.release()
cv2.destroyAllWindows()
