import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class FeatureMerge(nn.Module):
    """Feature merging branch"""
    def __init__(self, in_channels, out_channels):
        super(FeatureMerge, self).__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, 1)
        self.conv2 = ConvBNReLU(out_channels, out_channels, 3, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class EAST(nn.Module):
    def __init__(self, pretrained=True, feature_size=128):
        super(EAST, self).__init__()        
        # Backbone network (VGG16 for this implementation)
        vgg16 = models.vgg16(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(vgg16.features.children())[:23])
        
        # Feature extraction
        self.conv1 = nn.Conv2d(512, feature_size, 1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Feature merging branch
        self.merge1 = FeatureMerge(feature_size, feature_size)
        self.merge2 = FeatureMerge(feature_size, feature_size)
        self.merge3 = FeatureMerge(feature_size, feature_size)
        
        # Output branches
        self.score_output = nn.Sequential(
            nn.Conv2d(feature_size, 1, 1),
            nn.Sigmoid()
        )
        
        # 8 coordinates (4 points with x,y)
        self.geo_output = nn.Sequential(
            nn.Conv2d(feature_size, 8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Feature extraction from backbone
        features = self.backbone(x)

        # Upscale features and merge
        h, w = features.size(2), features.size(3)
        
        # Initial feature processing
        features = self.relu1(self.bn1(self.conv1(features)))
        
        # Merge features and upscale
        up1 = F.interpolate(features, size=(h*2, w*2), mode='bilinear', align_corners=True)
        features = self.merge1(up1)
        
        up2 = F.interpolate(features, size=(h*4, w*4), mode='bilinear', align_corners=True)
        features = self.merge2(up2)
        
        up3 = F.interpolate(features, size=(h*8, w*8), mode='bilinear', align_corners=True)
        features = self.merge3(up3)
        
        # Score map output
        score_map = self.score_output(features)
        # Geo map output
        geometry = self.geo_output(features)
        
        return score_map, geometry
    

class TextDetector(nn.Module):
    def __init__(self, pretrained=True, feature_size=128):
        super(TextDetector, self).__init__()        
        # Backbone network (VGG16 for this implementation)
        vgg16 = models.vgg16(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(vgg16.features.children())[:23])
        
        # Feature extraction
        self.conv1 = nn.Conv2d(512, feature_size, 1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Feature merging branch
        self.merge1 = FeatureMerge(feature_size, feature_size)
        self.merge2 = FeatureMerge(feature_size, feature_size)
        self.merge3 = FeatureMerge(feature_size, feature_size)
        
        # Output branches
        self.score_output = nn.Sequential(
            nn.Conv2d(feature_size, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Feature extraction from backbone
        features = self.backbone(x)
        
        # Upscale features and merge
        h, w = features.size(2), features.size(3)
        
        # Initial feature processing
        features = self.relu1(self.bn1(self.conv1(features)))
        
        # Merge features and upscale
        up1 = F.interpolate(features, size=(h*2, w*2), mode='bilinear', align_corners=True)
        features = self.merge1(up1)

        up2 = F.interpolate(features, size=(h*4, w*4), mode='bilinear', align_corners=True)
        features = self.merge2(up2)

        up3 = F.interpolate(features, size=(h*8, w*8), mode='bilinear', align_corners=True)
        features = self.merge3(up3)
        
        # Score map output
        score_map = self.score_output(features)
        return score_map