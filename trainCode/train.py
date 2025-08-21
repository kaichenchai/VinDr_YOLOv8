from ultralytics import YOLO, settings

settings.update({"wandb":True})

dataset = "data.yaml"
model = YOLO(f"yolo11m-obb.pt")

model.train(project = "pneumothorax-obb",
            data = dataset,
            name = "210825_yolov11m_512",
            batch = 32,
            epochs = 500,
            imgsz = 512,
            plots = True,
            device=[0],
            optimizer="Adam",
            patience=100,
            seed=12048)

