import wandb

if __name__ == "__main__":
    run = wandb.init(project="cardiomegaly_explainability", id="fk5pn8hy")
    wandb.save(glob_str="/home/kai/mnt/VinDr_YOLOv8_experiments/trainCode/cardiomegaly_explainability/04072025_yolov11m_histEQ/weights/*",
            base_path="/home/kai/mnt/VinDr_YOLOv8_experiments/trainCode/cardiomegaly_explainability/04072025_yolov11m_histEQ/")
    run.finish()
