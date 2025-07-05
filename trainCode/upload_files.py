import wandb

if __name__ == "__main__":
    run = wandb.init(project="cardiomegaly_explainability", id="h51kiaxk")
    wandb.save(glob_str="/home/kai/mnt/VinDr_Code/VinDr_YOLOv8/trainCode/cardiomegaly_explainability/05072025_yolov11l_histEQ3/weights/*",
            base_path="/home/kai/mnt/VinDr_Code/VinDr_YOLOv8/trainCode/cardiomegaly_explainability/05072025_yolov11l_histEQ3/")
    run.finish()
