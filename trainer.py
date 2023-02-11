import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from utils import get_criterion, get_lr_scheduler, get_optimizer

class Trainer:
    def __init__(self, device, model, dataloader, configs):
        self.device = device
        self.model = model
        self.dataloader = dataloader
        self.configs = configs

    def run(self):
        # Criterion
        criterion = get_criterion(self.configs)

        # Optimizer
        optimizer = get_optimizer(self.model.parameters(), self.configs)

        # LR Scheduler
        lr_scheduler = get_lr_scheduler(optimizer, self.configs)

        self.model = self.model.to(self.device)

        for epoch in range(self.configs.epochs):
            print(f"Epoch : {epoch + 1} ==========")

            # Train
            self.model.train()
            train_loss = 0.0
            n_samples = len(self.dataloader["train"].dataset)
            for inputs, labels in tqdm(self.dataloader["train"]):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                n_samples_batch = labels.shape[0]

                outputs = self.model(inputs)

                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lr_scheduler.step()

                train_loss += loss.items() * n_samples_batch
            
            print(f"Train Loss : {train_loss / n_samples}")

            # Eval
            self.model.eval()
            eval_loss = 0.0
            n_samples = len(self.dataloader["val"].dataset)
            for inputs, labels in tqdm(self.dataloader["val"]):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                n_samples_batch = labels.shape[0]

                with torch.no_grad():
                    outputs = self.model(inputs)

                    loss = criterion(outputs, labels)
                
                eval_loss += loss.items() * n_samples_batch
            
            print(f"Eval Loss : {eval_loss / n_samples}")

            wandb.log({
                "epoch" : epoch,
                "train loss" : train_loss,
                "eval loss" : eval_loss,
            })
        
        torch.save(self.model.state_dict(), self.configs.save_file)