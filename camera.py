import os
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model_path = os.path.join('.', 'runs', 'detect', 'train4', 'weights', 'last.pt')
model = YOLO(model_path)  # Replace with your model path

# Initialize video capture object
cap = cv2.VideoCapture(0)  # Change index if using external camera

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Perform object detection on the frame
    results = model(frame)

    # Loop through the detections
    for result in results:
        for box in result.boxes:
            coordinates = box.xyxy[0]
            x_min, y_min, x_max, y_max = int(coordinates[0]), int(coordinates[1]), int(coordinates[2]), int(
                coordinates[3])
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            confidence = box.conf[0]
            class_name = ""
            if box.cls[0] == 0 or box.cls[0] == 2:
                class_name = "Fresh"
            else:
                class_name = "Rotten"
            cv2.putText(frame, f"{class_name} {confidence:.2f}", (int(x_min), int(y_min) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0), 2)
    # Display the resulting frame
    cv2.imshow('YOLOv8 Object Detection', frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
