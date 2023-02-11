from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ImageNetDataLoader:
    def __init__(self, configs):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.dataloader = {
            "train": DataLoader(
                dataset=datasets.ImageNet(
                    root="data",
                    split="train",
                    transform=transform,
                ),
                batch_size=configs.batch_size,
                shuffle=True,
            ),
            "val": DataLoader(
                dataset=datasets.ImageNet(
                    root="data",
                    split="val",
                    transform=transform,
                ),
                batch_size=configs.batch_size,
                shuffle=False,
            ),
        }

    def __getitem__(self, train_or_eval):
        return self.dataloader[train_or_eval]