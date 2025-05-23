from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
import wandb

# Step 1: Initialize a Weights & Biases run
wandb.init(project="train_VinDr_YOLOv8", job_type="validation", name = "230525_YOLOv8m_subset-C-merged-CLAHE-bone-suppression-pretrained-ssl_VAL",
config={
    "dataset": "FULL_1024_padding_CLAHE_bone_suppression",
    "model": "YOLOv8m_last.pt",
    "image_size": 512,
    "batch": 16,
    "machine": "RTX4090",
    "conf": 0.20
}
)
best = "/mnt/data/kai/VinDr_Code/VinDr_YOLOv8/trainCode/train_VinDr_YOLOv8/220525_YOLOv8m_subset-C-merged-CLAHE-bone-suppression-pretrained-ssl5/weights/last.pt"
dataset_name = "subset.yaml"

#load the best of the trained model
model = YOLO(best)

#so gets tracked
add_wandb_callback(model, enable_model_checkpointing=True)

#validation info
metrics = model.val(project = "train_VinDr_YOLOv8",
            data = dataset_name,
            name = "230525_YOLOv8m_subset-C-merged-CLAHE-bone-suppression-pretrained-ssl_VAL",
            imgsz = 512,
            plots = True,
            batch = 16,
            save_json = True,
            device=[0],
            split = "test",
            max_det = 1,
            conf = 0.20)



#finishing run
wandb.finish()
