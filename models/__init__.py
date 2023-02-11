from .resnet import Resnet18

def get_model(configs):
    if configs.name == "resnet18":
        return Resnet18(configs)