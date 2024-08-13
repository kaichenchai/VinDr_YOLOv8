from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
import wandb

# Step 1: Initialize a Weights & Biases run
wandb.init(project="VinDr_YOLOv8", job_type="training", name = "TEST_YOLOv8_5000t600v_2e",
config={
    "epochs": 2,
    "dataset": "VinDr 5000train 600val",
    "model": "YOLOv8n",
    "image_size": 1024,
    "batch_size": 16,
    "machine": "Thermaltake_2080ti_0"
}
)
# Step 2: Define the YOLOv8 Model and Dataset
model_name = "yolov8n"
dataset_name = "model.yaml"
model = YOLO(f"{model_name}.pt")

# Step 3: Add W&B Callback for Ultralytics
add_wandb_callback(model, enable_model_checkpointing=True)

# Step 4: Train and Fine-Tune the Model
model.train(project = "VinDr_YOLOv8",
            data = dataset_name,
            name = "TEST_YOLOv8_5000t600v_2e",
            batch = 16,
            epochs = 2,
            imgsz = 1024,
            plots = True,
            device=[0],
            )

# Step 5: Validate the Model
try:
  model.val(data = "model.yaml", imgsz = 1024)
except AssertionError as e:
  print(f"Error Excepted: e")

# Step 7: Finalize the W&B Run
wandb.finish()
