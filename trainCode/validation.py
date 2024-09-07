from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
import wandb

# Step 1: Initialize a Weights & Biases run
wandb.init(project="VinDr_YOLOv8", job_type="validation", name = "200824_YOLOv8s_brightnessEQ_FIXED_FIXEDBB_VAL",
config={
    "dataset": "FULL_brightnessEQ_VinDr_FIXED",
    "model": "YOLOv8s_best.pt",
    "image_size": 1024,
    "batch": 16,
    "machine": "Thermaltake_2080ti_0"
}
)
best = "/mnt/data/kai/VinDr_YOLOv8_experiments/trainCode/train_VinDr_YOLOv8/200824_YOLOv8s_brightnessEQ_FIXED_FIXEDBB2/weights/best.pt"
dataset_name = "model.yaml"

#load the best of the trained model
model = YOLO(best)

#so gets tracked
add_wandb_callback(model, enable_model_checkpointing=True)

#validation info
metrics = model.val(project = "train_VinDr_YOLOv8",
            data = dataset_name,
            name = "200824_YOLOv8s_brightnessEQ_FIXED_FIXEDBB_VAL",
            imgsz = 1024,
            plots = True,
            ba`tch = 16,
            save_json = True,
            device=[0, 1])

#finishing run
wandb.finish()
