from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
import wandb

# Step 1: Initialize a Weights & Biases run
wandb.init(project="train_VinDr_YOLOv8", job_type="validation", name = "23092024_YOLOv8m_subset-C-merged_VAL_1DET",
config={
    "dataset": "FULL_brightnessEQ_VinDr_FIXED",
    "model": "YOLOv8m_best.pt",
    "image_size": 1024,
    "batch": 16,
    "machine": "Thermaltake_2080ti_0",
    "conf": 0.01
}
)
best = "/mnt/data/kai/VinDr_YOLOv8_experiments/trainCode/train_VinDr_YOLOv8/12092024_YOLOv8m_subset-C-merged/weights/best.pt"
dataset_name = "subset.yaml"

#load the best of the trained model
model = YOLO(best)

#so gets tracked
add_wandb_callback(model, enable_model_checkpointing=True)

#validation info
metrics = model.val(project = "train_VinDr_YOLOv8",
            data = dataset_name,
            name = "23092024_YOLOv8m_subset-C-merged_VAL1DET",
            imgsz = 1024,
            plots = True,
            batch = 16,
            save_json = True,
            device=[0,1],
            split = "test",
            max_det = 1,
            conf = 0.01)



#finishing run
wandb.finish()
