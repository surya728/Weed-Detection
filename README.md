# Weed Detection Using YOLO with Semi-Supervised Learning

This repository contains the implementation of a weed detection model leveraging YOLO, semi-supervised learning techniques, and a combination of labeled and unlabeled data. The goal is to detect and localize weeds in agricultural fields, reducing herbicide use and improving crop yield prediction.

## Dataset
- *Labeled Dataset*: 200 images of sesame crops and weeds.
- *Unlabeled Dataset*: 1000 similar images.
- *Test Dataset*: 100 images with annotations.

## Approach
The approach combines supervised learning using labeled data and semi-supervised learning using unlabeled data to enhance model performance. We applied the following techniques:

### 1. Data Preprocessing & Augmentation:
- *Rotation*: 90°, 180°, and 270° to diversify weed orientations.
- *Horizontal Flipping*: To increase generalization.
- *Bounding Box Adjustments*: After augmentations to maintain correct localization.

### 2. YOLO11m Training:
- *Training*: 50 epochs using labeled data with a batch size of 16, learning rate of 5e-5, and image size of 640x640.
- *GPU Acceleration*: Using CUDA for faster training.

### 3. Semi-Supervised Learning:
- *Consistency Regularization*: Applied weak and strong augmentations to ensure model stability.
- *Pseudo Labeling*: Generated pseudo labels for unlabeled data based on high-confidence predictions.
- *FixMatch Algorithm*: Used pseudo-labeled data for further model training.
- *Mean Teacher Model*: Utilized a Teacher model with EMA to guide learning.

### 4. Inference:
- *Test Inference*: Performed inference with a confidence threshold of 0.5 to detect and localize weeds.
- *Visualization*: Bounding boxes drawn around detected weeds and crops for qualitative assessment.

## Evaluation Metrics:
The performance of the model was evaluated using:
- *Precision* and *Recall* for detecting weeds.
- *F1-Score* to measure the harmonic mean of precision and recall.
- *Mean Average Precision (mAP@[.5:.95])* to evaluate the model across different IoU thresholds.
- *Combined Metric*: 0.5 * (F1-Score) + 0.5 * (mAP@[.5:.95]).

## Results:
The model achieved strong performance with the following:
- *High precision and recall* in weed detection.
- *Lower mAP scores* due to discrepancies between ground truth labels and model predictions.
- *Improved weed localization* and detection, even with partial or missing ground truth annotations.

## Challenges:
- *Discrepancy in Ground Truth Labels*: The model's predictions were often more accurate in terms of localization than the ground truth, leading to lower precision and recall but better detection.
- The model demonstrated the potential for *improving annotation quality* in future datasets by identifying missing or misclassified weeds.

## Inference:
The final model was evaluated on a set of test images, where it successfully detected and localized weeds. Bounding boxes closely aligned with ground truth, and the model performed well under varying environmental conditions.

## Model Performance Comparison

Below is a comparison of the performance metrics for different model variations during training and testing:

| *Model*                       | *Precision* | *Recall* | *mAP50* | *mAP50-95* | *Fitness* | *F1 Score* | *Combined Metric* |
|----------------------------------|---------------|------------|-----------|--------------|-------------|--------------|---------------------|
| *Base Model*                   | 0.8619        | 0.7655     | 0.8455    | 0.5005       | 0.5350      | 0.8108       | 0.6557              |
| *Consistency Regularization*   | 0.8078        | 0.8481     | 0.8409    | 0.5150       | 0.5476      | 0.8275       | 0.6712              |
| *Pseudo Labeling*              | 0.8027        | 0.8212     | 0.8408    | 0.5157       | 0.5482      | 0.8118       | 0.6637              |
| *FixMatch*                     | 0.8681        | 0.8010     | 0.8541    | 0.5392       | 0.5707      | 0.8332       | 0.6862              |
| *Mean Teacher (100 epochs)*    | 0.8426        | 0.8596     | 0.8747    | 0.5755       | 0.6054      | 0.8510       | 0.7133              |
| *Mean Teacher (200 epochs)*    | 0.9175        | 0.8505     | 0.9451    | 0.6956       | 0.7206      | 0.8827       | 0.7892              |

## Conclusion:
The combination of supervised and semi-supervised learning techniques in this weed detection model demonstrates its capability for real-world agricultural applications, providing a reliable method for weed localization and identification.

## Requirements:
- Python 3.x
- PyTorch
- YOLO11m (Ultralytics)
- OpenCV
- CUDA (for GPU acceleration)

## Usage:
```bash
# Clone the repository
git clone https://github.com/your-repo/weed-detection.git

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py --data dataset.yaml --epochs 50 --batch-size 16 --img-size 640

# Run inference on test images
python infer.py --weights path/to/model.pt --source path/to/test/images
