from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
import wandb

# Step 1: Initialize a Weights & Biases run
wandb.init(project="VinDr_YOLOv8", job_type="training", name = "03092024_YOLOv8s_FIXEDBB",
config={
    "epochs": 50,
    "dataset": "FULL_brightnessEQ_VinDr_FIXED_FIXEDBB",
    "model": "YOLOv8s",
    "image_size": 1024,
    "batch_size": 16,
    "machine": "Thermaltake_2080ti_0"
}
)
# Step 2: Define the YOLOv8 Model and Dataset
model_name = "yolov8s"
dataset_name = "model.yaml"
model = YOLO(f"{model_name}.pt")

# Step 3: Add W&B Callback for Ultralytics
add_wandb_callback(model, enable_model_checkpointing=True)

# Step 4: Train and Fine-Tune the Model
model.train(project = "train_VinDr_YOLOv8",
            data = dataset_name,
            name = "03092024_YOLOv8s_FIXEDBB",
            batch = 16,
            epochs = 50,
            imgsz = 1024,
            plots = True,
            device=[0])

# Step 7: Finalize the W&B Run
wandb.finish()
