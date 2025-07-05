from ultralytics import YOLO, settings

settings.update({"wandb":True})

dataset = "data.yaml"
model = YOLO(f"yolo11s.pt")

model.train(project = "cardiomegaly_explainability",
            data = dataset,
            name = "05072025_yolov11s_histEQ",
            batch = 16,
            epochs = 500,
            imgsz = 1024,
            plots = True,
            device=[0],
            optimizer="Adam",
            patience=100)

