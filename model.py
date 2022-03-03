import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import timm

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.resnet = torchvision.models.resnet18(pretrained = True)
        self.resnet.fc = nn.Linear(in_features = 512, out_features = num_classes, bias = True)
    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.resnet(x)
        return x
    
class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.resnet50 = torchvision.models.resnet50(pretrained = True)
        self.resnet50.fc = nn.Linear(in_features = 2048, out_features = num_classes, bias = True)
        torch.nn.init.xavier_uniform_(self.resnet50.fc.weight)
        stdv = 1. / math.sqrt(self.resnet50.fc.weight.size(1))
        self.resnet50.fc.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.resnet50(x)
        return x
    
class ResNet152(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.resnet152 = torchvision.models.resnet152(pretrained = True)
        self.resnet152.fc = nn.Linear(in_features = 2048, out_features = num_classes, bias = True)
        torch.nn.init.xavier_uniform_(self.resnet152.fc.weight)
        stdv = 1. / math.sqrt(self.resnet152.fc.weight.size(1))
        self.resnet152.fc.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.resnet152(x)
        return x
    
class tf_efficientnet_b7_ns(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.tf_efficientnet_b7_ns = timm.create_model('tf_efficientnet_b7_ns', pretrained = True, num_classes=num_classes)

        
#         torch.nn.init.xavier_uniform_(self.resnet152.fc.weight)
#         stdv = 1. / math.sqrt(self.resnet152.fc.weight.size(1))
#         self.resnet152.fc.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.tf_efficientnet_b7_ns(x)
        return x
    
class resnetrs420(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnetrs420 = timm.create_model('resnetrs420', pretrained = True, num_classes=num_classes)

        
#         torch.nn.init.xavier_uniform_(self.resnet152.fc.weight)
#         stdv = 1. / math.sqrt(self.resnet152.fc.weight.size(1))
#         self.resnet152.fc.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.resnetrs420(x)
        return x

class vit_small_r26_s32_384(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit_large_patch16_384 = timm.create_model('vit_small_r26_s32_384', pretrained = True, num_classes=num_classes)

        
#         torch.nn.init.xavier_uniform_(self.resnet152.fc.weight)
#         stdv = 1. / math.sqrt(self.resnet152.fc.weight.size(1))
#         self.resnet152.fc.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.vit_large_patch16_384(x)
        return x
    
class vit_base_patch16_384(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit_base_patch16_384 = timm.create_model('vit_base_patch16_384', pretrained = True, num_classes=num_classes)

        
#         torch.nn.init.xavier_uniform_(self.resnet152.fc.weight)
#         stdv = 1. / math.sqrt(self.resnet152.fc.weight.size(1))
#         self.resnet152.fc.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.vit_base_patch16_384(x)
        return x

class vit_large_patch16_224(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit_large_patch16_224 = timm.create_model('vit_large_patch16_224', pretrained = True, num_classes=num_classes)

        
#         torch.nn.init.xavier_uniform_(self.resnet152.fc.weight)
#         stdv = 1. / math.sqrt(self.resnet152.fc.weight.size(1))
#         self.resnet152.fc.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.vit_large_patch16_224(x)
        return x

class tf_efficientnet_b5_ns(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.tf_efficientnet_b5_ns = timm.create_model('tf_efficientnet_b5_ns', pretrained = True, num_classes=num_classes)

        
#         torch.nn.init.xavier_uniform_(self.resnet152.fc.weight)
#         stdv = 1. / math.sqrt(self.resnet152.fc.weight.size(1))
#         self.resnet152.fc.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.tf_efficientnet_b5_ns(x)
        return x


class swin_small_224(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('swin_small_patch4_window7_224', pretrained=True, num_classes=18)
    
    def forward(self, x):
        x = self.model(x)
        return x

class swin_large_224(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('swin_large_patch4_window7_224', pretrained=True, num_classes=18)
    
    def forward(self, x):
        x = self.model(x)
        return x

class swin_large_384(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('swin_large_patch4_window12_384', pretrained=True, num_classes=18)
    
    def forward(self, x):
        x = self.model(x)
        return x

class swin_base_patch4_384(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('swin_base_patch4_window12_384', pretrained=True, num_classes=18)
    
    def forward(self, x):
        x = self.model(x)
        return x

class efficientnetb4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=18)
    
    def forward(self, x):
        x = self.model(x)
        return x

class beit_base_patch16_384(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('beit_base_patch16_384', pretrained=True, num_classes=18)
    
    def forward(self, x):
        x = self.model(x)
        return x