import sys
import os
sys.path.append(os.getcwd())

import torch
from torch import nn
import torchsummary

from models.layers.conv_block import Conv2dBnRelu
from models.initialize import weight_initialize


class _Darknet19(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(_Darknet19, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # config out_channels, kernel_size
        stem = [
            [32, 3]
        ]
        layer1 = [
            'M',
            [64, 3]
        ]
        layer2 = [
            'M',
            [128, 3],
            [64, 1],
            [128, 3]
        ]
        layer3 = [
            'M',
            [256, 3],
            [128, 1],
            [256, 3]
        ]
        layer4 = [
            'M',
            [512, 3],
            [256, 1],
            [512, 3],
            [256, 1],
            [512, 3]
        ]
        layer5 = [
            'M',
            [1024, 3],
            [512, 1],
            [1024, 3],
            [512, 1],
            [1024, 3],
        ]

        self.stem = self._make_layers(stem)
        self.layer1 = self._make_layers(layer1)
        self.layer2 = self._make_layers(layer2)
        self.layer3 = self._make_layers(layer3)
        self.layer4 = self._make_layers(layer4)
        self.layer5 = self._make_layers(layer5)

        self.dropout = nn.Dropout2d(p=0.5)

        self.classifier = nn.Sequential(
            Conv2dBnRelu(1024, num_classes, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.dropout(x)
        
        x = self.classifier(x)
        return x

    def _make_layers(self, layer_cfg):
        layers = []

        for cfg in layer_cfg:
            if cfg == 'M':
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers.append(Conv2dBnRelu(self.in_channels, cfg[0], cfg[1]))
                self.in_channels = cfg[0]

        return nn.Sequential(*layers)


def darknet19(num_classes=1000, in_channels=3):
    model = _Darknet19(num_classes, in_channels)
    weight_initialize(model)
    return model


if __name__ == '__main__':
    input_size = 64
    in_channels = 3
    num_classes = 200

    tmp_input = torch.randn(1, 3, 64, 64)
    
    model = darknet19(num_classes, in_channels)
    
    torchsummary.summary(model, (in_channels, input_size, input_size), batch_size=1, device='cpu')
    
    from torchvision.models import vgg16_bn
    model = vgg16_bn(pretrained=False, num_classes=200)

    torchsummary.summary(model, (in_channels, input_size, input_size), batch_size=1, device='cpu')
    
    # with torch.no_grad():
    #     pred = model(tmp_input)
    # print(pred.size())
    # print(pred[0][0])


    # Load Weights from ckpt by pytorch-lightning
    # ckpt_path = os.path.join(os.getcwd(), 'saved/darknet19_tiny-imagenet-v2/version_0/checkpoints/epoch=249-step=97749.ckpt')
    # checkpoint = torch.load(ckpt_path)

    # state_dict = checkpoint["state_dict"]
    # for key in list(state_dict):
    #     state_dict[key.replace("model.", "")] = state_dict.pop(key)

    # model.load_state_dict(state_dict)

    # with torch.no_grad():
    #     pred = model(tmp_input)
    # print(pred.size())
    # print(pred[0][0])    
