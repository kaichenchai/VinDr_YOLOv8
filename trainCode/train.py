from ultralytics import YOLO, settings

settings.update({"wandb":True})

dataset = "data.yaml"
model = YOLO(f"yolo11l.pt")

model.train(project = "cardiomegaly_explainability",
            data = dataset,
            name = "05072025_yolov11l_histEQ",
            batch = 8,
            epochs = 500,
            imgsz = 1024,
            plots = True,
            device=[0],
            optimizer="Adam",
            patience=100)

