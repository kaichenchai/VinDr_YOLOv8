from ultralytics import YOLO, settings

settings.update({"wandb":True})

dataset = "data.yaml"
model = YOLO(f"yolo11m.pt")

model.train(project = "cardiomegaly_explainability",
            data = dataset,
            name = "06072025_yolov11m_CLAHE",
            batch = 16,
            epochs = 500,
            imgsz = 1024,
            plots = True,
            device=[0],
            optimizer="Adam",
            patience=100)

