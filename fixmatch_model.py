import os
import torch
import cv2
import torchvision.transforms as T
from ultralytics import YOLO

model = YOLO("model_pseudo.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.model.to(device)
model.model.train()

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

    weak_img_tensor = T.ToTensor()(img_pil).unsqueeze(0).to(device)
    weak_preds = model.predict(weak_img_tensor)

    if isinstance(weak_preds, list) and len(weak_preds) > 0:
        weak_preds = weak_preds[0]
    else:
        continue

    strong_img = strong_transform(img_pil)
    strong_tensor = T.ToTensor()(strong_img).unsqueeze(0).to(device).requires_grad_(True)

    strong_preds = model.predict(strong_tensor)
    if isinstance(strong_preds, list) and len(strong_preds) > 0:
        strong_preds = strong_preds[0]
    else:
        continue

    # FixMatch loss (only for high-confidence predictions)
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    for w_box, s_box in zip(weak_preds.boxes.data, strong_preds.boxes.data):
        cls_w, xw, yw, ww, hw, conf_w = w_box.tolist()
        cls_s, xs, ys, ws, hs, conf_s = s_box.tolist()

        if conf_w > 0.85:
            pseudo_label = torch.tensor([cls_w, xw, yw, ww, hw], device=device)
            pred_label = torch.tensor([cls_s, xs, ys, ws, hs], device=device)

            bbox_loss = torch.nn.functional.mse_loss(pred_label[1:], pseudo_label[1:])
            class_loss = torch.nn.functional.cross_entropy(pred_label[:1].unsqueeze(0),
                                                           pseudo_label[:1].unsqueeze(0).long())

            loss += bbox_loss + class_loss

    if loss.item() > 0:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.save("model_fixmatch.pt")