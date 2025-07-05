from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
import wandb

wandb.init(project="train_VinDr_YOLOv8", job_type="training", name = "220525_YOLOv8m_subset-C-merged-CLAHE-bone-suppression-no-pretrained",
config={
    "epochs": 100,
    "dataset": "FULL_1024_CLAHE_padding",
    "model": "YOLOv8m",
    "image_size": 512,
    "batch_size": 16,
    "machine": "RTX4090",
    "optimizer": "Adam",
})


# Step 2: Define the YOLOv8 Model and Dataset
model_name = "yolov8m"
dataset_name = "subset.yaml"
model = YOLO(f"{model_name}.pt")


# Step 3: Add W&B Callback for Ultralytics
add_wandb_callback(model, enable_model_checkpointing=True)


# Step 4: Train and Fine-Tune the Model
model.train(project = "train_VinDr_YOLOv8",
            data = dataset_name,
            name = "230525_YOLOv8m_subset-C-merged-CLAHE-bone-suppression-no-pretrained",
            epochs = 100,
            pretrained=False,
            batch = 16,
            imgsz = 512,
            plots = True,
            device=[0],
            optimizer="Adam"
            )

wandb.finish()
