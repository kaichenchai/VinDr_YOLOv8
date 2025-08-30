from ultralytics import YOLO, settings

settings.update({"wandb":True})

dataset = "data.yaml"
model = YOLO("yolo11m-obb.pt")

model.train(project = "pneumothorax",
            data = dataset,
            name = "290825_yolov11m_512_CLAHE_BS_obb",
            batch = 64,
            epochs = 200,
            imgsz = 512,
            plots = True,
            device=[0],
            optimizer="Adam",
            patience=50,
            seed=12048)

