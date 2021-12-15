import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, kernel_size=7, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1)//2, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size=7, stride=1, downsample=None, drop_block_size=None, input_length=5000):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        if drop_block_size is not None:
            self.dropout = DropBlock1D(drop_prob=0.2, block_size=drop_block_size)
        else:
            self.dropout = nn.Dropout(.2)

        downsample_ratio_0 = int((input_length + 3) / 4)
        downsample_ratio_1 = int((downsample_ratio_0 + 1) / 2)
        downsample_ratio_2 = int((downsample_ratio_1 + 1) / 2)
        downsample_ratio_3 = int((downsample_ratio_2 + 1) / 2)
        if planes == 64:
            self.globalAvgPool = nn.AvgPool1d(downsample_ratio_0, stride=1)
        elif planes == 128:
            self.globalAvgPool = nn.AvgPool1d(downsample_ratio_1, stride=1)
        elif planes == 256:
            self.globalAvgPool = nn.AvgPool1d(downsample_ratio_2, stride=1)
        elif planes == 512:
            self.globalAvgPool = nn.AvgPool1d(downsample_ratio_3, stride=1)

        self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 16))
        self.fc2 = nn.Linear(in_features=round(planes / 16), out_features=planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        original_out = out
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1)
        out = out * original_out
        out += residual
        out = self.relu(out)

        return out

# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm1d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm1d(planes)
#         self.downsample = downsample
#         self.stride = stride
#         self.dropout = nn.Dropout(.2)
#
#         self.globalAvgPool = nn.AdaptiveAvgPool1d(1)
#
#         self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 16))
#         self.fc2 = nn.Linear(in_features=round(planes / 16), out_features=planes)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.dropout(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         original_out = out
#         out = self.globalAvgPool(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc1(out)
#         out = self.relu(out)
#         out = self.fc2(out)
#         out = self.sigmoid(out)
#         out = out.view(out.size(0), out.size(1), 1)
#         out = out * original_out
#         out += residual
#         out = self.relu(out)
#
#         return out


# class SE_ResNet(nn.Module):
#     def __init__(self, layers=[3, 4, 6, 3], num_classes=26, channel_num=8):
#         super(SE_ResNet, self).__init__()
#         block = BasicBlock
#         self.inplanes = 64
#         self.external = 15
#         self.conv1 = nn.Conv1d(channel_num, 64, kernel_size=15, stride=2, padding=7, bias=False)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#         # self.fc = nn.Linear(512 * block.expansion + self.external, num_classes)
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm1d(planes * block.expansion), )
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#
#         x = self.fc(x)
#         return x

class SE_ResNet(nn.Module):
    def __init__(self, layers=[3,4,6,3], num_classes=26, kernel_size=7, channel_num=8, drop_block_size=None, input_length=5000):
        super(SE_ResNet, self).__init__()
        block = BasicBlock
        self.input_length = input_length
        self.inplanes = 64
        self.external = 0
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(channel_num, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], drop_block_size=drop_block_size)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, drop_block_size=drop_block_size)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, drop_block_size=drop_block_size)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, drop_block_size=drop_block_size)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion + self.external, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, drop_block_size=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion), )
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample,kernel_size=self.kernel_size, drop_block_size=drop_block_size, input_length=self.input_length))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=self.kernel_size, input_length=self.input_length))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # x3 = torch.cat([x, x2], dim=1)
        x = self.relu(x)
        x4 = self.fc(x)
        return x4

def se_resnet(layers=[3,4,6,3], num_classes=26, channel_num=8, drop_block_size=None, input_length=4992, kernel_size=7):
    model = SE_ResNet(layers=layers, num_classes=num_classes, channel_num=channel_num, drop_block_size=drop_block_size, kernel_size=kernel_size, input_length=input_length)
    return model


if __name__ == '__main__':
    model = se_resnet(layers=[3, 4, 6, 3], num_classes=26)
    # x = torch.randn((16, 12, 3000))
    x = torch.randn((16, 8, 4992))
    y = model(x)
    print(y.shape)
