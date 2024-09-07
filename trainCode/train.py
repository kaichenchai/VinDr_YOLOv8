from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
import wandb

wandb.require("core")

# Step 2: Define the YOLOv8 Model and Dataset
model_name = "yolov8m"
dataset_name = "model.yaml"
model = YOLO(f"{model_name}.pt")

wandb.init(project="VinDr_YOLOv8", job_type="training", group = "07092024_YOLOv8s_FIXEDBB",
config={
    "epochs": 50,
    "dataset": "FULL_brightnessEQ_VinDr_FIXED_FIXEDBB",
    "model": "YOLOv8m",
    "image_size": 1024,
    "batch_size": 16,
    "machine": "Thermaltake_2080ti_0"
},
name = "loading model"
)


wandb.init(project="VinDr_YOLOv8", job_type="training", group = "07092024_YOLOv8s_FIXEDBB",
config={
    "epochs": 50,
    "dataset": "FULL_brightnessEQ_VinDr_FIXED_FIXEDBB",
    "model": "YOLOv8m",
    "image_size": 1024,
    "batch_size": 16,
    "machine": "Thermaltake_2080ti_0"
},
name = "train"
)

# Step 3: Add W&B Callback for Ultralytics
add_wandb_callback(model, enable_model_checkpointing=True)


# Step 4: Train and Fine-Tune the Model
model.train(project = "train_VinDr_YOLOv8",
            data = dataset_name,
            name = "07092024_YOLOv8s_FIXEDBB",
            batch = 16,
            epochs = 1,
            imgsz = 1024,
            plots = True,
            device=[0, 1])

try:
    wandb.init(project="VinDr_YOLOv8", job_type="training", group = "07092024_YOLOv8s_FIXEDBB",
    config={
        "epochs": 50,
        "dataset": "FULL_brightnessEQ_VinDr_FIXED_FIXEDBB",
        "model": "YOLOv8m",
        "image_size": 1024,
        "batch_size": 16,
        "machine": "Thermaltake_2080ti_0"
    },
        name = "validation on test set"
    )

    model.val(project = "train_VinDr_YOLOv8",
            data = dataset_name,
            name = "VAL_07092024_YOLOv8s_FIXEDBB",
            imgsz = 1024,
            plots = True,
            batch = 16,
            save_json = True,
            device=[0, 1],
            split = "test")
except Exception as e:
    print(e)


# Step 7: Finalize the W&B Run
wandb.finish()
