import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
import math


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



# class TimmEfficientNetB3(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
        
#         self.model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=num_classes)
#         # for param in self.model.parameters():
#         #     param.require_grads = False
#         in_features = self.model.classifier.in_features # 1536
#         self.model.classifier = nn.Sequential(
#             nn.BatchNorm1d(in_features), 
#             nn.Linear(in_features=in_features, out_features=512, bias=False), 
#             nn.ReLU(), 
#             nn.BatchNorm1d(512), 
#             nn.Dropout(0.5), 
#             nn.Linear(in_features=512, out_features=num_classes, bias=False)
#             # nn.Linear(in_features=in_features, out_features=num_classes, bias=False)
#         )

#     def forward(self, x):
#         x = self.model.forward(x)
#         return x

class TimmEfficientNetB3_v1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=num_classes)
        # for param in self.model.parameters():
        #     param.require_grads = False
        in_features = self.model.classifier.in_features # 1536
        self.model.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features), 
            # nn.Linear(in_features=in_features, out_features=512, bias=False), 
            # nn.ReLU(), 
            # nn.BatchNorm1d(512), 
            nn.Dropout(0.5), 
            # nn.Linear(in_features=512, out_features=num_classes, bias=False)
            nn.Linear(in_features=in_features, out_features=num_classes, bias=False)
        )


    def forward(self, x):
        x = self.model.forward(x)
        return x


class TimmEfficientNetB4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=num_classes)
        # for param in self.model.parameters():
        #     param.require_grads = False
        in_features = self.model.classifier.in_features # 1536
        self.model.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features), 
            nn.Dropout(0.5), 
            nn.Linear(in_features=in_features, out_features=num_classes, bias=False)
        )


    def forward(self, x):
        x = self.model.forward(x)
        return x

# class TimmEfficientNetB3_v2(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
        
#         self.model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=num_classes)
#         # for param in self.model.parameters():
#         #     param.require_grads = False
#         in_features = self.model.classifier.in_features # 1536
#         self.model.classifier = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

#         torch.nn.init.xavier_uniform_(self.model.classifier.weight)
#         stdv = 1. / math.sqrt(self.model.classifier.weight.size(1))
#         self.model.classifier.bias.data.uniform_(-stdv, stdv)


#     def forward(self, x):
#         x = self.model.forward(x)
#         return x



# swin_base_patch4_window12_384
class TimmSwinBasePatch4Window12_384(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.model = timm.create_model('swin_base_patch4_window12_384', pretrained=True, num_classes=num_classes)
        # for param in self.model.parameters():
        #     param.require_grads = False


    def forward(self, x):
        x = self.model.forward(x)
        return x



# swin_large_patch4_window12_384
class TimmSwinLargePatch4Window12_384(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.model = timm.create_model('swin_large_patch4_window12_384', pretrained=True, num_classes=num_classes)
        # for param in self.model.parameters():
        #     param.require_grads = False


    def forward(self, x):
        x = self.model.forward(x)
        return x




class TVResNext50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = models.resnext50_32x4d(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, out_features=num_classes, bias=True)
        torch.nn.init.xavier_uniform_(self.model.fc.weight)
        stdv = 1. / math.sqrt(self.model.fc.weight.size(1))
        self.model.fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.model.forward(x)
        return x



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



if __name__ == "__main__":
    model = TVResNext50(num_classes=18)
    print(model)