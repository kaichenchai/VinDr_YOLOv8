from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
import wandb

wandb.require("core")

# Step 2: Define the YOLOv8 Model and Dataset
dataset_name = "model.yaml"
model = YOLO("yolov8m.pt")

# Step 3: Add W&B Callback for Ultralytics
add_wandb_callback(model, enable_model_checkpointing=True)

# Step 4: Train and Fine-Tune the Model
model.train(project = "train_VinDr_YOLOv8",
            data = dataset_name,
            name = "24042025_YOLOv8m_FULL_1024_CLAHE_VinDr_padding",
            batch = 16,
            epochs = 50,
            imgsz = 1024,
            plots = True,
            device=[0])

wandb.finish()
