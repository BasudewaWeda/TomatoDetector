import os
import cv2
import time
import requests
import threading
from ultralytics import YOLO

# Load the YOLOv8 model
model_path = os.path.join('.', 'runs', 'detect', 'train4', 'weights', 'last.pt')
model = YOLO(model_path)  # Replace with your model path

# Initialize video capture object
cap = cv2.VideoCapture(0)  # Change index if using external camera

# Define a delay between POST requests (e.g., 1 second)
post_delay = 5.0
last_post_time = time.time()

# Frame skipping parameter
frame_skip = 8
frame_count = 0

# Shared resources
frame_lock = threading.Lock()
frame = None


def process_frames():
    global frame, last_post_time, frame_count
    while True:
        with frame_lock:
            if frame is None:
                continue
            current_frame = frame.copy()

        # Perform object detection on the frame
        results = model(current_frame)

        # Process the detection results
        for result in results:
            for box in result.boxes:
                coordinates = box.xyxy[0]
                x_min, y_min, x_max, y_max = int(coordinates[0]), int(coordinates[1]), int(coordinates[2]), int(
                    coordinates[3])
                cv2.rectangle(current_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                confidence = box.conf[0]
                class_name = ""
                if box.cls[0] == 0 or box.cls[0] == 2:
                    class_name = "Fresh"
                else:
                    class_name = "Rotten"
                cv2.putText(current_frame, f"{class_name} {confidence:.2f}", (int(x_min), int(y_min) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Send POST request if enough time has passed since the last request
                current_time = time.time()
                if current_time - last_post_time >= post_delay:
                    data = {'type': class_name.lower()}
                    try:
                        requests.post('http://localhost:5000/update', json=data)
                    except requests.exceptions.RequestException as e:
                        print(f"Error sending POST request: {e}")
                    last_post_time = current_time

        # Display the resulting frame
        cv2.imshow('YOLOv8 Object Detection', current_frame)

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Start the frame processing thread
processing_thread = threading.Thread(target=process_frames)
processing_thread.daemon = True
processing_thread.start()

while True:
    # Capture frame-by-frame
    ret, current_frame = cap.read()
    frame_count += 1

    if frame_count % frame_skip == 0:
        with frame_lock:
            frame = current_frame

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
