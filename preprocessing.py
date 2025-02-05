import os
import cv2
import numpy as np
import shutil
import random
from pathlib import Path

def rotate_image(image, angle):
    """Rotates the image by the given angle."""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def rotate_bbox(bbox, angle):
    """Rotates bounding box coordinates."""
    class_id, x_center, y_center, width, height = bbox

    if angle == 90:
        new_x = y_center
        new_y = 1 - x_center
    elif angle == 180:
        new_x = 1 - x_center
        new_y = 1 - y_center
    elif angle == 270:
        new_x = 1 - y_center
        new_y = x_center
    else:
        return bbox

    return [class_id, new_x, new_y, width, height]

def flip_image(image):
    """Flips the image horizontally."""
    return cv2.flip(image, 1)

def flip_bbox(bbox):
    """Flips bounding box coordinates horizontally."""
    class_id, x_center, y_center, width, height = bbox
    new_x_center = 1 - x_center
    return [class_id, new_x_center, y_center, width, height]

def create_augmented_dataset():
    """Creates an augmented dataset (rotated, flipped images)."""
    base_path = Path("/kaggle/input/new-dataset/labeled")
    output_path = Path("/kaggle/working/augmented_dataset")

    for subdir in ['images', 'annotations']:
        (output_path / subdir).mkdir(parents=True, exist_ok=True)

    for img_path in (base_path / "images").glob("*.jpg"):
        ann_path = base_path / "annotations" / f"{img_path.stem}.txt"

        if not ann_path.exists():
            continue

        shutil.copy2(img_path, output_path / "images" / img_path.name)
        shutil.copy2(ann_path, output_path / "annotations" / ann_path.name)

        image = cv2.imread(str(img_path))
        with open(ann_path, 'r') as f:
            bboxes = [list(map(float, line.strip().split())) for line in f]

        for angle in [90, 180, 270]:
            new_img_name = f"{img_path.stem}_rot{angle}.jpg"
            new_ann_name = f"{img_path.stem}_rot{angle}.txt"

            rotated_image = rotate_image(image, angle)
            rotated_bboxes = [rotate_bbox(bbox, angle) for bbox in bboxes]

            cv2.imwrite(str(output_path / "images" / new_img_name), rotated_image)
            with open(output_path / "annotations" / new_ann_name, 'w') as f:
                for bbox in rotated_bboxes:
                    f.write(' '.join(map(str, bbox)) + '\n')

        new_img_name = f"{img_path.stem}_flip.jpg"
        new_ann_name = f"{img_path.stem}_flip.txt"

        flipped_image = flip_image(image)
        flipped_bboxes = [flip_bbox(bbox) for bbox in bboxes]

        cv2.imwrite(str(output_path / "images" / new_img_name), flipped_image)
        with open(output_path / "annotations" / new_ann_name, 'w') as f:
            for bbox in flipped_bboxes:
                f.write(' '.join(map(str, bbox)) + '\n')

    print(f"✅ Augmented dataset saved in: {output_path}")


def restructure_dataset():
    """Moves the augmented dataset into Train, Dev, and Unlabeled folders."""
    ORIGINAL_DATASET_PATH = Path("/kaggle/input/new-dataset")
    AUGMENTED_DATASET_PATH = Path("/kaggle/working/augmented_dataset")
    OUTPUT_DATASET_PATH = Path("/kaggle/working/kriti-25-final")

    TRAIN_PATH = OUTPUT_DATASET_PATH / "Dataset" / "Train"
    DEV_PATH = OUTPUT_DATASET_PATH / "Dataset" / "Dev"
    UNLABELED_PATH = OUTPUT_DATASET_PATH / "Dataset" / "Unlabelled"

    for subdir in ["images", "annotations"]:
        (TRAIN_PATH / subdir).mkdir(parents=True, exist_ok=True)
        (DEV_PATH / subdir).mkdir(parents=True, exist_ok=True)
        UNLABELED_PATH.mkdir(parents=True, exist_ok=True)

    for subdir in ["images", "annotations"]:
        src_path = AUGMENTED_DATASET_PATH / subdir
        dest_path = TRAIN_PATH / subdir

        for file in src_path.glob("*"):
            shutil.move(str(file), str(dest_path / file.name))

    shutil.copytree(ORIGINAL_DATASET_PATH / "test", DEV_PATH, dirs_exist_ok=True)
    shutil.copytree(ORIGINAL_DATASET_PATH / "unlabeled", UNLABELED_PATH, dirs_exist_ok=True)
    print("Dataset restructuring complete!")


def prepare_yolo_dataset():
    """Prepares YOLO dataset structure for training."""
    BASE_DIR = Path("/kaggle/working/kriti-25-final/Dataset")
    TRAIN_IMG_DIR = BASE_DIR / "Train" / "images"
    TRAIN_ANN_DIR = BASE_DIR / "Train" / "annotations"
    DEV_IMG_DIR = BASE_DIR / "Dev" / "images"
    DEV_ANN_DIR = BASE_DIR / "Dev" / "annotations"

    NEW_DATASET_DIR = Path("/kaggle/working/yolo_dataset")
    IMG_TRAIN_DIR = NEW_DATASET_DIR / "images" / "train"
    IMG_VAL_DIR = NEW_DATASET_DIR / "images" / "val"
    LBL_TRAIN_DIR = NEW_DATASET_DIR / "labels" / "train"
    LBL_VAL_DIR = NEW_DATASET_DIR / "labels" / "val"

    for folder in [IMG_TRAIN_DIR, IMG_VAL_DIR, LBL_TRAIN_DIR, LBL_VAL_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(TRAIN_IMG_DIR.glob("*.jpg"))) + sorted(list(DEV_IMG_DIR.glob("*.jpg")))
    random.seed(42)
    random.shuffle(image_files)

    split_idx = int(0.8 * len(image_files))
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    def move_files(image_list, img_dest, lbl_dest, src_img_dir, src_lbl_dir):
        for img_path in image_list:
            img_name = img_path.name
            lbl_name = img_name.replace(".jpg", ".txt")
            lbl_path = Path(src_lbl_dir) / lbl_name

            if lbl_path.exists():
                shutil.copy(img_path, img_dest)
                shutil.copy(lbl_path, lbl_dest)

    move_files(train_files, IMG_TRAIN_DIR, LBL_TRAIN_DIR, TRAIN_IMG_DIR, TRAIN_ANN_DIR)
    move_files(val_files, IMG_VAL_DIR, LBL_VAL_DIR, TRAIN_IMG_DIR, TRAIN_ANN_DIR)
    move_files(train_files, IMG_TRAIN_DIR, LBL_TRAIN_DIR, DEV_IMG_DIR, DEV_ANN_DIR)
    move_files(val_files, IMG_VAL_DIR, LBL_VAL_DIR, DEV_IMG_DIR, DEV_ANN_DIR)

    yaml_content = f""" train: {IMG_TRAIN_DIR}
                        val: {IMG_VAL_DIR}
                        nc: 2
                        names: ["Crop", "Weed"]
                    """

    with open(NEW_DATASET_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)

    print("YOLO dataset ready for training.")

if __name__ == "__main__":
    create_augmented_dataset()
    restructure_dataset()
    prepare_yolo_dataset()
    print("All dataset preparation steps completed successfully!")