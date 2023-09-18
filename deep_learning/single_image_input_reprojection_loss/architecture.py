import torchvision
import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F


import torchvision
import torch.nn as nn
import torch
import torchvision.models as models





class UNet(nn.Module):
    def __init__(self, in_channels=3, drop_rate=0.1):
        super().__init__()

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_rate),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_rate)
            )

        self.dconv_down1 = double_conv(in_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last_mask1 = nn.Conv2d(64, 1, 1)
        self.conv_last_mask2 = nn.Conv2d(64, 1, 1)



        #freezing pretrained weights
        #non_frozen_layers = ["dconv_up1", "conv_last_mask1", "conv_last_mask2"]
        non_frozen_layers = []

        for name, param in self.named_parameters():
            if not any(layer in name for layer in non_frozen_layers):
                param.requires_grad = False


    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)
        x = self.upsample(x)
        
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)

        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)

        x = torch.cat([x, conv1], dim=1)
        features = self.dconv_up1(x)


        out_mask1 = torch.sigmoid(self.conv_last_mask1(features))
        out_mask2 = torch.sigmoid(self.conv_last_mask2(features))




        return out_mask1, out_mask2



class CameraParameterNetV2(nn.Module):
    def __init__(self, resnet_variant='resnet50', use_unet=True):
        super(CameraParameterNetV2, self).__init__() 
        
        resnet_classes = {
            'resnet18': models.resnet18,
            'resnet50': models.resnet50,
            'resnet152': models.resnet152
        }
        
        # Use specified ResNet as the backbone for feature extraction
        resnet_backbone = resnet_classes[resnet_variant](pretrained=False)
        
        input_channels = 5 if use_unet else 3
        # Modify the first convolutional layer based on whether UNet is used or not
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
        
        # Regressing 6 parameters
        self.fc3_6_params = nn.Linear(num_features//4, 6)
        
        self.relu = nn.ReLU()

    def forward(self, x, square_size):
        square_size = square_size.float()

        x = self.feature_extractor(x)
        x = torch.cat((x, square_size.unsqueeze(-1)), dim=1)  # concatenate the size
        x = self.fc1(self.relu(x))
        x = self.fc2(self.relu(x))
        
        # Extracting first 4 parameters
        params_4 = self.fc3_4_params(self.relu(x))
        
        # Extracting next 6 parameters
        params_6 = self.fc3_6_params(self.relu(x))
        
        return torch.cat((params_4, params_6), dim=1)
      
class CombinedModel(nn.Module):
    def __init__(self, resnet_variant='resnet50', use_unet=True):
        super(CombinedModel, self).__init__()

        self.use_unet = use_unet
        
        if self.use_unet:
            # UNet module
            self.unet = UNet()
            self.unet.eval()
        
        # Dual Input module
        self.camnet = CameraParameterNetV2(resnet_variant=resnet_variant, use_unet=self.use_unet)

    def forward(self, x, square_size):
        if self.use_unet:
            outmask1, outmask2 = self.unet(x)
            x = torch.cat([x, outmask1, outmask2], dim=1)
        
        params = self.camnet(x, square_size)
        return params