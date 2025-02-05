import os
import torch
import cv2
import torchvision.transforms as T
from ultralytics import YOLO

model = YOLO("model_consistency.pt")
UNLABELED_DIR = "/kaggle/working/kriti-25-final/Dataset/Unlabelled"
PSEUDO_LABELS_DIR = "/kaggle/working/yolo_dataset/labels/pseudo"
os.makedirs(PSEUDO_LABELS_DIR, exist_ok=True)

results = model.predict(source=UNLABELED_DIR, save=False)

for i, result in enumerate(results):
    img_name = os.path.basename(result.path).replace(".jpg", ".txt")
    label_path = os.path.join(PSEUDO_LABELS_DIR, img_name)
    with open(label_path, "w") as f:
        for box in result.boxes.data:
            cls, x_center, y_center, width, height, conf = box.tolist()
            if conf > 0.85:
                f.write(f"{int(cls)} {x_center} {y_center} {width} {height}\n")

TRAIN_IMAGES_DIR = "/kaggle/working/yolo_dataset/images/train"
TRAIN_LABELS_DIR = "/kaggle/working/yolo_dataset/labels/train"

for file in os.listdir(PSEUDO_LABELS_DIR):
    if file.endswith(".txt"):
        shutil.copy(os.path.join(PSEUDO_LABELS_DIR, file), TRAIN_LABELS_DIR)
model.train(data="/kaggle/working/data.yaml", epochs=50, imgsz=640, batch=16, device="cuda")

model.save("model_pseudo.pt")