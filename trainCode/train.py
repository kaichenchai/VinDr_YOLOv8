from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
import wandb

wandb.require("core")

# Step 2: Define the YOLOv8 Model and Dataset
model_name = "yolov8m"
dataset_name = "kaggle.yaml"
model = YOLO(f"{model_name}.pt")


# Step 3: Add W&B Callback for Ultralytics
add_wandb_callback(model, enable_model_checkpointing=True)


# Step 4: Train and Fine-Tune the Model
model.train(project = "train_VinDr_YOLOv8",
            data = dataset_name,
            name = "10092024_YOLOv8s_kaggleBB",
            epochs = 50,
            batch = 16,
            imgsz = 1024,
            plots = True,
            device=[0, 1]
            )

wandb.finish()
