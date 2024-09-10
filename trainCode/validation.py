from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
import wandb

# Step 1: Initialize a Weights & Biases run
wandb.init(project="train_VinDr_YOLOv8", job_type="validation", name = "10092024_YOLOv8m_kaggleBB_VAL",
config={
    "dataset": "FULL_brightnessEQ_VinDr_FIXED",
    "model": "YOLOv8m_best.pt",
    "image_size": 1024,
    "batch": 16,
    "machine": "Thermaltake_2080ti_0"
}
)
best = "/mnt/data/kai/VinDr_YOLOv8_experiments/trainCode/train_VinDr_YOLOv8/10092024_YOLOv8s_kaggleBB/weights/best.pt"
dataset_name = "kaggle.yaml"

#load the best of the trained model
model = YOLO(best)

#so gets tracked
add_wandb_callback(model, enable_model_checkpointing=True)

#validation info
metrics = model.val(project = "train_VinDr_YOLOv8",
            data = dataset_name,
            name = "10092024_YOLOv8m_kaggleBB_VAL",
            imgsz = 1024,
            plots = True,
            batch = 16,
            save_json = True,
            device=[0, 1],
            split = "test")

#finishing run
wandb.finish()
