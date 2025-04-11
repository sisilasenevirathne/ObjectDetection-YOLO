import cv2
from ultralytics import YOLO

# Load YOLOv8 model (nano version for speed)
model = YOLO('yolov8n.pt')

# Load video (you can change this to 0 for webcam)
cap = cv2.VideoCapture('D://SJAY/Python/Jupyter/Eggs/EggsTrain.mp4')
#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('rtsp://admin:Abc@1234@192.168.3.251/Streaming/Channels/1')

assert cap.isOpened(), "Error reading video file"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    results = model(frame)

    # Draw results on the frame
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
