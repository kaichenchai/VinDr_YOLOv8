import wandb

if __name__ == "__main__":
    run = wandb.init(project="cardiomegaly_explainability", id="fj00w39p")
    wandb.save(glob_str="/home/kai/mnt/VinDr_Code/VinDr_YOLOv8/trainCode/cardiomegaly_explainability/06072025_yolov11m_CLAHE2/weights/*",
            base_path="/home/kai/mnt/VinDr_Code/VinDr_YOLOv8/trainCode/cardiomegaly_explainability/06072025_yolov11m_CLAHE2/")
    run.finish()
