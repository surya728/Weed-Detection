import torch
import os
import shutil
from copy import deepcopy
from ultralytics import YOLO

# Load Student Model
student_model = YOLO("model_fixmatch.pt").to("cuda")
student_model.train()

# Initialize Teacher Model as an EMA of Student
teacher_model = deepcopy(student_model)
teacher_model.eval()

# Ensure pseudo-label directory exists
PSEUDO_LABELS_DIR = "/kaggle/working/labels/pseudo/"
os.makedirs(PSEUDO_LABELS_DIR, exist_ok=True)

# Define optimizer for student model
optimizer = torch.optim.AdamW(student_model.model.parameters(), lr=5e-5)

# EMA Decay Factor
alpha = 0.999

# Load Unlabeled Data
UNLABELED_DIR = "/kaggle/input/kriti-25-final/Augmented Dataset Dihing/Unlabelled"
image_files = [f for f in os.listdir(UNLABELED_DIR) if f.endswith(".jpg")]

# Train Student Model with EMA Teacher
for img_file in image_files:
    img_path = os.path.join(UNLABELED_DIR, img_file)

    # Get pseudo-labels from Teacher Model
    teacher_preds = teacher_model.predict(img_path, conf=0.85)

    if len(teacher_preds) == 0:
        continue  # Skip if no detections

    teacher_preds = teacher_preds[0]  # Extract first result

    # Convert pseudo-labels to training format
    label_file = os.path.join(PSEUDO_LABELS_DIR, img_file.replace(".jpg", ".txt"))
    with open(label_file, "w") as f:
        for i in range(len(teacher_preds.boxes.xywhn)):
            x_center, y_center, width, height = teacher_preds.boxes.xywhn[i].tolist()
            cls = int(teacher_preds.boxes.cls[i].item())
            f.write(f"{cls} {x_center} {y_center} {width} {height}\n")

print("✅ Pseudo-label generation completed.")

# Retrain the student model using pseudo-labels
student_model.train(data="/kaggle/working/data.yaml", epochs=50, imgsz=640, batch=16, device="cuda")

# Update Teacher Model with EMA after training
for teacher_param, student_param in zip(teacher_model.model.parameters(), student_model.model.parameters()):
    teacher_param.data = alpha * teacher_param.data + (1 - alpha) * student_param.data

# Save the Mean Teacher Student Model
student_model.save("model_mean_teacher.pt")

print("✅ Mean Teacher Model training completed and saved as 'model_mean_teacher.pt'.")