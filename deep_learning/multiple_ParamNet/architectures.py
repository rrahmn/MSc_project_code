import torchvision
import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F


import torchvision
import torch.nn as nn
import torch
import torchvision.models as models


class CameraParameterNetV2(nn.Module):
    def __init__(self, resnet_variant='resnet50'):
        super(CameraParameterNetV2, self).__init__() 
        
        resnet_classes = {
            'resnet18': models.resnet18,
            'resnet50': models.resnet50,
            'resnet152': models.resnet152
        }
        
        # Use specified ResNet as the backbone for feature extraction
        resnet_backbone = resnet_classes[resnet_variant](pretrained=False)
        
        input_channels = 15
        # Modify the first convolutional layer 
        resnet_backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the final layer (classification layer) of ResNet
        self.feature_extractor = nn.Sequential(*list(resnet_backbone.children())[:-1], nn.Flatten())
        
        # Get the number of out_features of the penultimate layer of ResNet
        num_features = resnet_backbone.fc.in_features
        
        # Define common feature layers
        self.fc1 = nn.Linear(num_features + 1, num_features//2) 
        self.fc2 = nn.Linear(num_features//2, num_features//4)
        
        # Regressing 4 parameters
        self.fc3_4_params = nn.Linear(num_features//4, 4)
        
        # Regressing 6 parameters for 5 images
        self.fc3_6_params = nn.Linear(num_features//4, 6*5)
        
        self.relu = nn.ReLU()

    def forward(self, x, square_size):
        square_size = square_size.float()

        x = self.feature_extractor(x)
        x = torch.cat((x, square_size.unsqueeze(-1)), dim=1)  # concatenate the size
        x = self.fc1(self.relu(x))
        x = self.fc2(self.relu(x))
        
        # Extracting first 4 parameters
        params_4 = self.fc3_4_params(self.relu(x))
        
        # Extracting next 6 parameters each for 5 images
        params_6 = self.fc3_6_params(self.relu(x))
        
        return torch.cat((params_4, params_6), dim=1)
      
class MutipledModel(nn.Module):
    def __init__(self, resnet_variant='resnet50'):
        super(MutipledModel, self).__init__()

        self.camnet = CameraParameterNetV2(resnet_variant=resnet_variant)

    def forward(self, x, square_size):
        params = self.camnet(x, square_size)
        return params