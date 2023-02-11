from .imagenet import ImageNetDataLoader

def get_dataloader(configs):
    if configs.name == "ImageNet":
        return ImageNetDataLoader(configs)