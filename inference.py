import cv2
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load trained YOLO model
model = YOLO("/kaggle/input/mean_teacher/pytorch/default/1/mean_teacher.pt").to("cuda")
model.eval()

# Path to test images
TEST_IMAGES_DIR = "/kaggle/working/yolo_dataset/images/val"
test_images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith(".jpg")]

# Run inference and visualize results
for img_file in test_images[20:40]:
    img_path = os.path.join(TEST_IMAGES_DIR, img_file)

    # Perform inference
    results = model.predict(img_path, conf=0.5)  # Set confidence threshold

    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct visualization

    # Plot results
    plt.figure(figsize=(10, 10))

    # Draw bounding boxes on the image
    for result in results:
        for box in result.boxes.xyxy.cpu().numpy():  # Convert tensor to numpy array
            x_min, y_min, x_max, y_max = map(int, box[:4])  # Extract bounding box coordinates
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green box

    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Detections for {img_file}")
    plt.show()