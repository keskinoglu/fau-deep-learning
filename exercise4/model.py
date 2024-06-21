import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        #self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        #self.bn1 = nn.BatchNorm2d(out_channels)
        #self.relu = nn.ReLU(inplace=True)
        #self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        #self.bn2 = nn.BatchNorm2d(out_channels)

        #self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1)
        #self.bn1x1 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        #residual = x
        #out = self.conv1(x)
        #out = self.bn1(out)
        #out = self.relu(out)
        #out = self.conv2(out)
        #out = self.bn2(out)
        #if (x.shape[1] != out.shape[1]) or (x.shape[2] != out.shape[2]):
        #    residual = nn.functional.conv2d(x, weight=nn.init.kaiming_normal_(torch.empty(out.shape[1], x.shape[1], 1, 1), mode='fan_out', nonlinearity='relu'), stride=self.stride)
        #out += residual
        #out = self.relu(out)
        #return out
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        #residual = F.conv2d(residual, x, stride=self.stride)
        residual = self.conv1x1(residual)
        residual = self.bn3(residual)

        x = x.add(residual)
        x = F.relu(x)

        return x

    
class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        #self.conv1 = torch.nn.Conv2d(3, 64, 7, 2)
        #self.bn1 = torch.nn.BatchNorm2d(64)
        #self.relu = torch.nn.ReLU()
        #self.pool = torch.nn.MaxPool2d(3, 2)
        #self.res_block1 = ResBlock(64, 64, 1)
        #self.res_block2 = ResBlock(64, 128, 2)
        #self.res_block3 = ResBlock(128, 256, 2)
        #self.res_block4 = ResBlock(256, 512, 2)
        #self.global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = torch.nn.Linear(512, 2)
        #self.sigmoid = torch.nn.Sigmoid()

        self.conv1 = nn.Conv2d(3, 64, 7, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.mpool = nn.MaxPool2d(3, 2)
        
        # ResBlock(in_channels, out_channels, stride):
        self.res_block1 = ResBlock(64, 64, 1)
        self.res_block2 = ResBlock(64, 128, 2)
        self.res_block3 = ResBlock(128, 256, 2)
        self.res_block4 = ResBlock(256, 512, 2)

        self.fc = nn.Linear(512, 2)
        self.flatten = nn.Flatten() # no trainable params but is not included in the nn.functional lib


    def forward(self, x):
        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.relu(x)
        #x = self.pool(x)
        #x = self.res_block1(x)
        #x = self.res_block2(x)
        #x = self.res_block3(x)
        #x = self.res_block4(x)
        #x = self.global_avg_pool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)
        #x = self.sigmoid(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.mpool(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = F.adaptive_avg_pool2d(x, 1) # global avg pooling
        x = self.flatten(x)
        x = self.fc(x)
        #x = F.sigmoid(x) # deprecated
        x = torch.sigmoid(x)

        return x