import torch
from torch import nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, drop_rate=0.1, fc_out_size=8):
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
        # self.conv_regression = nn.Conv2d(64, 1, 1)

        # self.spatial_reduction = nn.AdaptiveAvgPool2d((320, 320))
        # self.corner_fc1 = nn.Linear(320*320, 128)
        # self.corner_fc2 = nn.Linear(128, 64)
        # self.corner_fc3 = nn.Linear(64, fc_out_size)  # Predicting 8 values for 4 corners (x,y) each


        # self.corner_features = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Input channels set to 1
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        # )

        # self.corner_predictor = nn.Sequential(
        #     nn.Linear(64, 32),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(32, fc_out_size),
        # )

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

        
        # chessboardfeatures = self.conv_last_mask1(features)
        # out_mask1 = torch.sigmoid(chessboardfeatures)
        out_mask1 = torch.sigmoid(self.conv_last_mask1(features))
        out_mask2 = torch.sigmoid(self.conv_last_mask2(features))

        #corner_features = self.conv_regression(features)
        # corner_features = self.spatial_reduction(chessboardfeatures)
        # corner_features = corner_features.view(corner_features.size(0), -1)  # Flatten the tensor
        # corner_features = F.relu(self.corner_fc1(corner_features))
        # corner_features = F.relu(self.corner_fc2(corner_features))
        # corner_coordinates = self.corner_fc3(corner_features)


        # corner_features = self.corner_features(out_mask1)
        # corner_features = corner_features.view(corner_features.size(0), -1)

        # corner_coordinates= self.corner_predictor(corner_features)


        return out_mask1, out_mask2#, corner_coordinates