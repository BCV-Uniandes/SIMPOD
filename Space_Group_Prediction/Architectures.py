import torch.nn as nn
from torchvision.models import alexnet, resnet50, densenet121, swin_t, swin_v2_t
from torchvision.models import AlexNet_Weights, ResNet50_Weights, DenseNet121_Weights, Swin_T_Weights, Swin_V2_T_Weights

# --------------------------------- SG Architectures ------------------------------------
class AlexNet_SG(nn.Module):
    def __init__(self, pretrained = True):
        super().__init__()
        if pretrained:
            self.model = alexnet(weights = AlexNet_Weights.DEFAULT)
        else:
            self.model = alexnet()
        self.model.classifier[6] = nn.Linear(4096, 230)

    def forward(self, x):
        return(self.model(x))
    

class ResNet50_SG(nn.Module):
    def __init__(self, pretrained = True):
        super().__init__()
        if pretrained:
            self.model = resnet50(weights = ResNet50_Weights.DEFAULT)
        else:
            self.model = resnet50()
        self.model.fc = nn.Linear(2048, 230)

    def forward(self, x):
        return(self.model(x))
    

class DenseNet121_SG(nn.Module):
    def __init__(self, pretrained = True):
        super().__init__()
        if pretrained:
            self.model = densenet121(weights = DenseNet121_Weights.DEFAULT)
        else:
            self.model = densenet121()
        self.model.classifier = nn.Linear(1024, 230)
        
    def forward(self, x):
        return(self.model(x))
    

class SwinT_SG(nn.Module):
    def __init__(self, pretrained = True):
        super().__init__()
        if pretrained:
            self.model = swin_t(weights = Swin_T_Weights.DEFAULT)
        else:
            self.model = swin_t()
        self.model.head = nn.Linear(768, 230)

    def forward(self, x):
        return(self.model(x))
    

class SwinT_v2_SG(nn.Module):
    def __init__(self, pretrained = True):
        super().__init__()
        if pretrained:
            self.model = swin_v2_t(weights = Swin_V2_T_Weights.DEFAULT)
        else:
            self.model = swin_v2_t()
        self.model.head = nn.Linear(768, 230)

    def forward(self, x):
        return(self.model(x))

    
def model_selector(name, pretrained = True):
    if name == 'alexnet':
        model = AlexNet_SG(pretrained = pretrained)
    elif name == 'resnet':
        model = ResNet50_SG(pretrained = pretrained)
    elif name == 'densenet':
        model = DenseNet121_SG(pretrained = pretrained)
    elif name == 'swin':
        model = SwinT_SG(pretrained = pretrained)
    elif name == 'swinv2':
        model = SwinT_v2_SG(pretrained = pretrained)
    return model