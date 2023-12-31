import torch.nn as nn
import torch.nn.functional as F
from custom_resnet import ResBlock

# Drop out is kept to avoid overfitting
dropout_value_min = 0.03

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()

        # Prep Layer
        self.preplayer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_value_min),
        )
        
        # Residual Block 1
        self.res1 = ResBlock(128)
        
        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_value_min),
        )
        
        # Layer 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(dropout_value_min),
        )

        # Residual Block 2
        self.res2 = ResBlock(512)
        
        # Maxpool
        self.maxpool = nn.MaxPool2d(4)

        # FC Layer
        self.fc = nn.Linear(512, num_classes)
                
    def forward(self, x):
        out = self.preplayer(x)
        
        out = self.layer1(out)
        r1 = self.res1(out)
        out = out + r1

        out = self.layer2(out)

        out = self.layer3(out)
        r2 = self.res2(out)
        out = out + r2
        
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = F.log_softmax(out, dim=1) # we need to use log(softmax), as the cross entropy needs the log function
        return out
    

    