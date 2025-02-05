#evaluation.py file

import torchvision.transforms as T
import torch
import os
import cv2
from ultralytics import YOLO

# Load trained YOLO model
model = YOLO("/kaggle/input/mean_teacher/pytorch/default/1/mean_teacher.pt").to("cuda")
model.eval()

# Run model validation and get results
results = model.val(data="/kaggle/working/yolo_dataset/data.yaml")

# Extract result_dict
result_dict = results.results_dict  # This should contain all the evaluation metrics

# Print all metrics
print("Metrics from result_dict:")
for key, value in result_dict.items():
    print(f"{key}: {value}")

# Compute F1 Score: F1 = 2 * (Precision * Recall) / (Precision + Recall)
precision = result_dict.get('metrics/precision(B)', 0)  # Using .get() to avoid KeyError
recall = result_dict.get('metrics/recall(B)', 0)

if precision + recall > 0:
    f1_score = 2 * (precision * recall) / (precision + recall)
else:
    f1_score = 0

print(f"\nComputed F1 Score: {f1_score:.4f}")

# Compute Combined Metric: 0.5 * (F1-Score) + 0.5 * (mAP[.5:.95])
mAP_50_95 = result_dict.get('metrics/mAP50-95(B)', 0)  # Using .get() to ensure no errors

combined_metric = 0.5 * f1_score + 0.5 * mAP_50_95

print(f"Computed Combined Metric: {combined_metric:.4f}")