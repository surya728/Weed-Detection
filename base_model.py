from ultralytics import YOLO

model = YOLO("yolo11m.pt")
model.train(data="/kaggle/working/data.yaml", epochs=50, imgsz=640, batch=16, device="cuda")
model.save("v11m_50_epochs.pt")