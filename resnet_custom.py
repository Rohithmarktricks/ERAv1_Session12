import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule, Trainer
import torchmetrics
from torchmetrics.classification import Accuracy


class ResidualBlock(LightningModule):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
#         out += residual
        out = self.relu(out)
        return out
    

class CustomResnet2(LightningModule):
    
    def __init__(self, num_classes=10):
        super().__init__()

        # PrepLayer
        self.prep_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.res_block1 = ResidualBlock(128)

        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Layer 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.res_block2 = ResidualBlock(512)

        # Max Pooling with Kernel Size 4
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=1)

        # Fully Connected Layer
        self.fc = nn.Linear(512, num_classes)

        # Softmax
        self.softmax = nn.Softmax(dim=1)
        
        #  metrics and loss
        self.loss_obj = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task='multiclass', num_classes=10)
        self.valid_acc = Accuracy(task='multiclass', num_classes=10)

    def forward(self, x):
        x = self.prep_layer(x)
        x = self.layer1(x)
        x = self.res_block1(x) + x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.res_block2(x) + x
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "valid_loss",
            }
        }
    
    def _shared_step(self, batch, metrics, prefix):
        inputs, targets = batch
        outputs = self(inputs)
#         print(outputs)
        loss = self.loss_obj(outputs, targets)
        _, predicted = outputs.max(1)
        acc = metrics(predicted, targets)
        self.log(f'{prefix}_loss', loss)
        self.log(f'{prefix}_acc', acc)
        return {'loss': loss, 'acc': acc}
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, self.train_acc, 'train')
    
    def on_train_epoch_end(self):
        train_accuracy = self.train_acc.compute()
        self.log('epoch_train_accuracy', train_accuracy)
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, self.valid_acc, 'valid')
    
    def on_validation_epoch_end(self):
        valid_accuracy = self.valid_acc.compute()
        self.log('epoch_valid_accuracy', valid_accuracy)