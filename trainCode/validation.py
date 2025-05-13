from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
import wandb

# Step 1: Initialize a Weights & Biases run
wandb.init(project="train_VinDr_YOLOv8", job_type="validation", name = "120525_YOLOv8m_subset-C-merged-brightnessEQ_512_VAL",
config={
    "dataset": "FULL_1024_padding_brightnessEQ",
    "model": "YOLOv8m_best.pt",
    "image_size": 512,
    "batch": 16,
    "machine": "RTX4090",
    "conf": 0.10
}
)
best = "/mnt/data/kai/VinDr_Code/VinDr_YOLOv8/trainCode/train_VinDr_YOLOv8/12092024_YOLOv8m_subset-C-merged2/weights/last.pt"
dataset_name = "subset.yaml"

#load the best of the trained model
model = YOLO(best)

#so gets tracked
add_wandb_callback(model, enable_model_checkpointing=True)

#validation info
metrics = model.val(project = "train_VinDr_YOLOv8",
            data = dataset_name,
            name = "130525_YOLOv8m_subset-C-merged-brightnessEQ_512_VAL",
            imgsz = 512,
            plots = True,
            batch = 16,
            save_json = True,
            device=[0],
            split = "test",
            max_det = 1,
            conf = 0.10)



#finishing run
wandb.finish()
