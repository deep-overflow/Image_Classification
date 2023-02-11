import torch.nn as nn

class Resnet18(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            ResidualBlocks(64, 64, 256, 3, down=False),
            ResidualBlocks(256, 128, 512, 4),
            ResidualBlocks(512, 256, 1024, 6),
            ResidualBlocks(1024, 512, 2048, 3),
        )

        self.classifier = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=7,
                stride=1,
                padding=0,
            ),
            nn.Flatten(),
            nn.Linear(2048, configs.n_class)
        )

    def forward(self, inputs):
        hidden1 = self.layer1(inputs)
        hidden2 = self.layer2(hidden1)
        return self.classifier(hidden2)

class ResidualBlocks(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, depth, down=True):
        super().__init__()
        self.blocks = nn.Sequential(
            ResidualBlock(
                in_channels=in_channels,
                mid_channels=mid_channels,
                out_channels=out_channels,
                channels=True,
                down=down,
            )
        )

        for _ in range(depth - 1):
            self.blocks.append(ResidualBlock(
                in_channels=out_channels,
                mid_channels=mid_channels,
                out_channels=out_channels
            ))
    
    def forward(self, inputs):
        return self.blocks(inputs)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, channels=False, down=False):
        super().__init__()
        self.channels = channels
        if down:
            stride = 2
        else:
            stride = 1
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=out_channels),
        )
        
        if self.channels:
            self.residual = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
            )
        
        self.activation = nn.ReLU()

    def forward(self, inputs):
        outputs = self.block(inputs)
        if self.channels:
            inputs = self.residual(inputs)
        return self.activation(outputs + inputs)