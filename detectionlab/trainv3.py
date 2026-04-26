import os
import yaml
import torch
from ultralytics import RTDETR

# Optimization settings
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()

# Configuration
DATASET_ROOT = os.path.abspath("yolo_dataset_augmented")
YAML_PATH = os.path.join(DATASET_ROOT, "data.yaml")
NUM_EPOCHS = 100
IMGSZ = 1280
NAME = "run_full_final"

if __name__ == "__main__":
    if not os.path.exists(YAML_PATH):
        raise FileNotFoundError(f"YAML config not found at {YAML_PATH}")

    # Parse YAML for classes
    with open(YAML_PATH) as f:
        data_yaml = yaml.safe_load(f)
        
    nc = data_yaml.get("nc")
    if nc is None:
        raise ValueError("data.yaml must define 'nc'")

    class_names = data_yaml.get("names", [f"class_{i}" for i in range(nc)])
    if len(class_names) != nc:
        raise ValueError(f"len(names)={len(class_names)} does not match nc={nc}")

    # Filter out ignored classes
    ignored_class_names = {"text_block", "text_bloc", "text_content"}
    ignored_indices = {i for i, name in enumerate(class_names) if name in ignored_class_names}
    valid_indices = [i for i in range(nc) if i not in ignored_indices]

    print(f"Valid classes to train: {[class_names[i] for i in valid_indices]}")

    # Initialize model
    model = RTDETR("rtdetr-l.pt")
    #model = RTDETR("yolo_dataset_augmented/train_results/run_full_final2/weights/last.pt")
    # Start training
    print(f"Starting training with dataset: {DATASET_ROOT}")
    model.train(
        data=YAML_PATH,
        epochs=NUM_EPOCHS,
        imgsz=IMGSZ,
        batch=32,          
        workers=20,         
        device=[1],
        project=os.path.join(DATASET_ROOT, "train_results"),
        name=NAME,
        save_period=10,
        optimizer="AdamW",
        lr0=1e-4,
        cos_lr=False,
        deterministic=False,
        val=False,     
        classes=valid_indices,
        resume=False,
        conf=0.5,
    )