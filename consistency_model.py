import torchvision.transforms as T
import torch
import os
import cv2
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("v11m_50_epochs.pt")
model.model.to(device)

weak_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomResizedCrop(size=640, scale=(0.8, 1.0))
])

strong_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomResizedCrop(size=640, scale=(0.5, 1.0)),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))
])

UNLABELED_DIR = "/kaggle/working/kriti-25-final/Dataset/Unlabelled"
image_files = [f for f in os.listdir(UNLABELED_DIR) if f.endswith(".jpg")]

optimizer = torch.optim.Adam(model.model.parameters(), lr=5e-5)
for img_file in image_files:
    img_path = os.path.join(UNLABELED_DIR, img_file)
    img = cv2.imread(img_path)
    img_pil = T.ToPILImage()(img)

    weak_img = weak_transform(img_pil)
    strong_img = strong_transform(img_pil)
    weak_tensor = T.ToTensor()(weak_img).unsqueeze(0).to(device).requires_grad_(True)
    strong_tensor = T.ToTensor()(strong_img).unsqueeze(0).to(device).requires_grad_(True)

    model.model.to(device)
    weak_preds = model.model(weak_tensor)  # Get raw output logits
    strong_preds = model.model(strong_tensor)

    if isinstance(weak_preds, tuple):
        weak_preds = weak_preds[0]
        strong_preds = strong_preds[0]

    #Consistency loss
    loss = torch.nn.functional.mse_loss(weak_preds, strong_preds)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.save("model_consistency.pt")