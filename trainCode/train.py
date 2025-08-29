from ultralytics import YOLO, settings

settings.update({"wandb":True})

dataset = "data.yaml"
model = YOLO("yolo11m.pt")

model.train(project = "pneumothorax",
            data = dataset,
            name = "210825_yolov11m_512_bb",
            batch = 64,
            epochs = 500,
            imgsz = 512,
            plots = True,
            device=[0],
            optimizer="Adam",
            patience=100,
            seed=12048)

