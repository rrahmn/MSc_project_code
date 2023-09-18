import torchvision
import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F


import torchvision
import torch.nn as nn
import torch
import torchvision.models as models


class CornerNet(nn.Module):
    def __init__(self, resnet_variant='resnet50'):
        super(CornerNet, self).__init__() 
        
        resnet_classes = {
            'resnet18': models.resnet18,
            'resnet50': models.resnet50,
            'resnet152': models.resnet152
        }
        
        # Use specified ResNet as the backbone for feature extraction
        resnet_backbone = resnet_classes[resnet_variant](pretrained=False)
        
        input_channels = 3
        # Modify the first convolutional layer 
        resnet_backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the final layer (classification layer) of ResNet
        self.feature_extractor = nn.Sequential(*list(resnet_backbone.children())[:-1], nn.Flatten())
        
        # Get the number of out_features of the penultimate layer of ResNet
        num_features = resnet_backbone.fc.in_features
        
        # Define common feature layers
        self.fc1 = nn.Linear(num_features, num_features//2) 
        self.fc2 = nn.Linear(num_features//2, num_features//4)
        
        # Regressing 4 parameters
        self.corner1x = nn.Linear(num_features//4, 1)
        self.corner2x = nn.Linear(num_features//4, 1)
        self.corner3x = nn.Linear(num_features//4, 1)
        self.corner4x = nn.Linear(num_features//4, 1)
        self.corner1y = nn.Linear(num_features//4, 1)
        self.corner2y = nn.Linear(num_features//4, 1)
        self.corner3y = nn.Linear(num_features//4, 1)
        self.corner4y = nn.Linear(num_features//4, 1)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc1(self.relu(x))
        x = self.fc2(self.relu(x))
        
        corner1x = self.corner1x(self.relu(x))
        corner2x = self.corner2x(self.relu(x))
        corner3x = self.corner3x(self.relu(x))
        corner4x = self.corner4x(self.relu(x))
        corner1y = self.corner1y(self.relu(x))
        corner2y = self.corner2y(self.relu(x))
        corner3y = self.corner3y(self.relu(x))
        corner4y = self.corner4y(self.relu(x))
        
        
        return torch.cat((corner1x, corner1y, corner2x, corner2y, corner3x, corner3y, corner4x, corner4y), dim=1)