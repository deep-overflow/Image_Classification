import torch
from models import get_model
from dataloaders import get_dataloader
from trainer import Trainer
import wandb

import argparse
from omegaconf import OmegaConf

if __name__ == "__main__":
    # Config file
    parser = argparse.ArgumentParser(description="Arguments for Image Classification Models")
    parser.add_argument("--conf_file", type=str, default="basic.yaml")
    args = parser.parse_args()

    # Configs
    configs = OmegaConf.load("configs/" + args.conf_file)

    # Wandb
    wandb.init(
        project="Image Classification",
        config=configs,
    )

    # Device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    
    print(f"Device : {device}")

    # Dataloader
    dataloader = get_dataloader(configs.dataset)

    # Model
    model = get_model(configs.model)

    # Trainer
    trainer = Trainer(device, model, dataloader, configs.train)

    # Train
    trainer.run()