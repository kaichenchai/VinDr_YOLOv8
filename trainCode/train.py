from ultralytics import YOLO, settings

settings.update({"wandb":True})

dataset = "data.yaml"
model = YOLO(f"yolo11m.pt")

model.train(project = "cardiomegaly_explainability",
            data = dataset,
            name = "04072025_yolov11m_histEQ",
            batch = 16,
            epochs = 500,
            imgsz = 1024,
            plots = True,
            device=[0, 1],
            optimizer="Adam")

